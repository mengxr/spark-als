package als

import com.github.fommil.netlib.BLAS.{getInstance => blas}

import scala.collection.mutable.ArrayBuffer

import org.apache.spark.{Partitioner, HashPartitioner}
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD

class SimpleALS {

  import SimpleALS._

  def run(ratings: RDD[(Int, Int, Float)], k: Int = 10, numBlocks: Int = 10, numIterations: Int = 10): (RDD[(Int, Array[Float])], RDD[(Int, Array[Float])]) = {
    val userPart = new HashPartitioner(numBlocks)
    val prodPart = new HashPartitioner(numBlocks)
    val blockRatings = blockifyRatings(ratings, userPart, prodPart).cache()
    val (userInBlocks, userOutBlocks) = makeBlocks("user", blockRatings, userPart, prodPart)
    val swappedBlockRatings = blockRatings.map { case ((userBlockId, prodBlockId), RatingBlock(userIds, prodIds, localRatings)) =>
      ((prodBlockId, userBlockId), RatingBlock(prodIds, userIds, localRatings))
    }
    val (prodInBlocks, prodOutBlocks) = makeBlocks("prod", swappedBlockRatings, prodPart, userPart)
    var userFactors = initialize(userInBlocks, k)
    var prodFactors = initialize(prodInBlocks, k)
    for (iter <- 0 until numIterations) {
      prodFactors = computeFactors(userFactors, userOutBlocks, prodInBlocks, k)
      userFactors = computeFactors(prodFactors, prodOutBlocks, userInBlocks, k)
    }
    val userIdAndFactors = userInBlocks.mapValues(_.srcIds).join(userFactors).values.cache()
    val prodIdAndFactors = prodInBlocks.mapValues(_.srcIds).join(prodFactors).values.cache()
    userIdAndFactors.count()
    prodIdAndFactors.count()
    userInBlocks.unpersist()
    userOutBlocks.unpersist()
    prodInBlocks.unpersist()
    prodOutBlocks.unpersist()
    val userOutput = userIdAndFactors.flatMap { case (ids, factors) =>
      ids.view.zip(factors)
    }
    val prodOutput = prodIdAndFactors.flatMap { case (ids, factors) =>
      ids.view.zip(factors)
    }
    (userOutput, prodOutput)
  }
}

object SimpleALS {

  type FactorBlock = (Int, Array[Array[Float]])
  type OutBlock = (Int, Array[Array[Int]])

  /**
   * In links for computing src factors.
   * @param srcIds
   * @param dstPtrs
   * @param dstBlockIds
   * @param dstLocalIndices
   * @param ratings
   */
  case class InBlock(
    srcIds: Array[Int],
    dstPtrs: Array[Int],
    dstBlockIds: Array[Short],
    dstLocalIndices: Array[Int],
    ratings: Array[Float])

  def initialize(inBlocks: RDD[(Int, InBlock)], k: Int): RDD[FactorBlock] = {
    inBlocks.map { case (srcBlockId, inBlock) =>
      val random = new java.util.Random(srcBlockId)
      val factors = Array.fill(inBlock.srcIds.size) {
        val factor = Array.fill(k)(random.nextFloat())
        val nrm = blas.snrm2(k, factor, 1)
        blas.sscal(k, 1.0f / nrm, factor, 1)
        factor
      }
      (srcBlockId, factors)
    }
  }

  case class RatingBlock(srcIds: Array[Int], dstIds: Array[Int], localRatings: Array[Float])

  def blockifyRatings(ratings: RDD[(Int, Int, Float)], srcPart: Partitioner, dstPart: Partitioner): RDD[((Int, Int), RatingBlock)] = {
    ratings.groupBy(x => (srcPart.getPartition(x._1), dstPart.getPartition(x._2)))
        .mapPartitions({ iter =>
      iter.map { case ((srcBlockId, dstBlockId), entries) =>
        val srcIds = ArrayBuffer.empty[Int]
        val dstIds = ArrayBuffer.empty[Int]
        val localRatings = ArrayBuffer.empty[Float]
        entries.foreach { case (srcId, dstId, rating) =>
          srcIds += srcId
          dstIds += dstId
          localRatings += rating
        }
        ((srcBlockId, dstBlockId), RatingBlock(srcIds.toArray, dstIds.toArray, localRatings.toArray))
      }
    }, preservesPartitioning = true).setName("blockRatings")
  }

  def makeBlocks(
      prefix: String,
      ratingBlocks: RDD[((Int, Int), RatingBlock)],
      srcPart: Partitioner,
      dstPart: Partitioner): (RDD[(Int, InBlock)], RDD[OutBlock]) = {
    val inBlocks = ratingBlocks.map { case ((srcBlockId, dstBlockId), RatingBlock(srcIds, dstIds, localRatings)) =>
      val dstIdToLocalIndex = dstIds.toSet.toSeq.sorted.zipWithIndex.toMap
      val dstLocalIndices = dstIds.map(dstIdToLocalIndex.apply)
      (srcBlockId, (dstBlockId, srcIds, dstLocalIndices, localRatings))
    }.groupByKey(new IdentityPartitioner(srcPart.numPartitions))
        .mapValues { iter =>
      // TODO: use better sort
      val entries = iter.flatMap { case (dstBlockId, theSrcIds, theDstLocalIndices, theLocalRatings) =>
        (0 until theSrcIds.size).map { i =>
          (theSrcIds(i), dstBlockId.toShort, theDstLocalIndices(i), theLocalRatings(i))
        }
      }.toSeq.sorted
      val size = entries.size
      val srcIds = ArrayBuffer.empty[Int]
      val dstCounts = ArrayBuffer.empty[Int]
      val dstBlockIds = new Array[Short](size)
      val dstLocalIndices = new Array[Int](size)
      val localRatings = new Array[Float](size)
      var i = 0
      var j = -1
      entries.foreach { case (srcId, dstBlockId, dstLocalIndex, rating) =>
        if (srcIds.isEmpty || srcId != srcIds.last) {
          srcIds += srcId
          dstCounts += 0
          j += 1
        }
        dstCounts(j) += 1
        dstBlockIds(i) = dstBlockId
        dstLocalIndices(i) = dstLocalIndex
        localRatings(i) = rating
        i += 1
      }
      val dstPtrs = dstCounts.scanLeft(0)(_ + _).toArray
      InBlock(srcIds.toArray, dstPtrs, dstBlockIds, dstLocalIndices, localRatings)
    }.setName(prefix + "InBlocks").cache()
    val outBlocks = inBlocks.mapValues { case InBlock(srcIds, dstPtrs, dstBlockIds, _, _) =>
      val activeIds = Array.fill(dstPart.numPartitions)(ArrayBuffer.empty[Int])
      var i = 0
      while (i < srcIds.size) {
        var j = dstPtrs(i)
        var preDstBlockId: Short = -1
        while (j < dstPtrs(i + 1)) {
          val dstBlockId = dstBlockIds(j)
          if (dstBlockId != preDstBlockId) {
            activeIds(dstBlockId) += i
            preDstBlockId = dstBlockId
          }
          j += 1
        }
        i += 1
      }
      activeIds.map(_.toArray)
    }.setName(prefix + "OutBlocks").cache()
    (inBlocks, outBlocks)
  }

  def computeFactors(
      srcFactorBlocks: RDD[FactorBlock],
      srcOutBlocks: RDD[OutBlock],
      dstInBlocks: RDD[(Int, InBlock)],
      k: Int): RDD[FactorBlock] = {
    val srcOut = srcOutBlocks.join(srcFactorBlocks).flatMap { case (srcBlockId, (srcOutBlock, srcFactors)) =>
      srcOutBlock.view.zipWithIndex.map { case (activeIndices, dstBlockId) =>
        (dstBlockId, (srcBlockId, activeIndices.map(idx => srcFactors(idx))))
      }
    }
    val merged = srcOut.groupByKey(new IdentityPartitioner(dstInBlocks.partitions.size))
    dstInBlocks.join(merged).mapValues { case (InBlock(dstIds, srcPtrs, srcBlockIds, srcLocalIndices, ratings), srcFactors) =>
      val sortedSrcFactors = srcFactors.toSeq.sortBy(_._1).map(_._2).toArray
      val dstFactors = new Array[Array[Float]](dstIds.size)
      var j = 0
      while (j < dstIds.size) {
        val ls = new LeastSquares(k)
        var i = srcPtrs(j)
        while (i < srcPtrs(j + 1)) {
          val srcBlockId = srcBlockIds(i)
          val srcLocalIndex = srcLocalIndices(i)
          ls.add(sortedSrcFactors(srcBlockId)(srcLocalIndex), ratings(i))
          i += 1
        }
        dstFactors(j) = ls.solve(lambda = 0.1f)
        j += 1
      }
      dstFactors
    }
  }
}
