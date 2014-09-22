package als

import com.github.fommil.netlib.BLAS.{getInstance => blas}

import scala.collection.mutable.ArrayBuffer

import org.apache.spark.{Partitioner, HashPartitioner}
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD

import cern.colt.map.OpenIntObjectHashMap

class SimpleALS {

  import SimpleALS._

  def run(ratings: RDD[(Int, Int, Float)], k: Int = 10, numBlocks: Int = 10, numIterations: Int = 10): (RDD[FactorBlock], RDD[FactorBlock]) = {
    val userPart = new HashPartitioner(numBlocks)
    val prodPart = new HashPartitioner(numBlocks)
    val (userInBlocks, userOutBlocks) = makeBlocks("user", ratings, userPart, prodPart)
    val (prodInBlocks, prodOutBlocks) = makeBlocks("prod", ratings.map(x => (x._2, x._1, x._3)), prodPart, userPart)
    var userFactors = initialize(userInBlocks, k)
    var prodFactors = initialize(prodInBlocks, k)
    for (iter <- 0 until numIterations) {
      prodFactors = computeFactors(userFactors, userOutBlocks, prodInBlocks, k)
      userFactors = computeFactors(prodFactors, prodOutBlocks, userInBlocks, k)
    }
    userFactors.cache()
    prodFactors.cache()
    userFactors.count()
    prodFactors.count()
    userInBlocks.unpersist()
    userOutBlocks.unpersist()
    prodInBlocks.unpersist()
    prodOutBlocks.unpersist()
    (userFactors, prodFactors)
  }
}

object SimpleALS {

  type FactorBlock = Array[(Int, Array[Float])]
  type OutBlock = Array[Array[Int]]

  /**
   * In links for computing src factors.
   * @param srcIds
   * @param dstPtrs
   * @param dstIds
   * @param ratings
   */
  case class InBlock(
    srcIds: Array[Int],
    dstPtrs: Array[Int],
    dstIds: Array[Int],
    ratings: Array[Float]) {

    def foreach(f: (Int, Int, Float) => Unit): Unit = {
      var i = 0
      while (i < srcIds.length) {
        val srcId = srcIds(i)
        var j = dstPtrs(i)
        while (j < dstPtrs(i + 1)) {
          f(srcId, dstIds(j), ratings(j))
          j += 1
        }
        i += 1
      }
    }
  }

  def initialize(inBlocks: RDD[InBlock], k: Int): RDD[FactorBlock] = {
    inBlocks.mapPartitionsWithIndex  { (srcBlockId, iter) =>
      val random = new java.util.Random(srcBlockId)
      val inBlock = iter.next()
      Iterator.single(inBlock.srcIds.map { srcId =>
        val factor = Array.fill(k)(random.nextFloat())
        val nrm = blas.snrm2(k, factor, 1)
        blas.sscal(k, 1.0f/nrm, factor, 1)
        (srcId, factor)
      })
    }
  }

  def makeBlocks(
      prefix: String,
      ratings: RDD[(Int, Int, Float)],
      srcPart: Partitioner,
      dstPart: Partitioner): (RDD[InBlock], RDD[OutBlock]) = {
    val inBlocks = ratings.map(x => (x._1, (x._2, x._3)))
        .groupByKey(srcPart)
        .mapPartitionsWithIndex { (srcBlockId, iter) =>
      val srcIds = ArrayBuffer.empty[Int]
      val dstCounts = ArrayBuffer.empty[Int]
      val dstIds = ArrayBuffer.empty[Int]
      val theRatings = ArrayBuffer.empty[Float]
      iter.foreach { case (srcId, dstIdAndRatings) =>
        srcIds += srcId
        var count = 0
        dstIdAndRatings.foreach { case (dstId, rating) =>
          count += 1
          dstIds += dstId
          theRatings += rating
        }
        dstCounts += count
      }
      val dstPtrs = dstCounts.toArray.scanLeft(0)(_ + _)
      Iterator.single(InBlock(srcIds.toArray, dstPtrs, dstIds.toArray, theRatings.toArray))
    }
    inBlocks.setName(prefix + "InBlocks").cache()
    val outBlocks = inBlocks.map { inBlock =>
      val sets = Array.fill(srcPart.numPartitions)(ArrayBuffer.empty[Int])
      inBlock.foreach { case (srcId, dstId, rating) =>
        sets(dstPart.getPartition(dstId)) += srcId
      }
      sets.map(_.toArray)
    }
    outBlocks.setName(prefix + "OutBlocks").cache()
    (inBlocks, outBlocks)
  }

  def computeFactors(
      srcFactorBlocks: RDD[FactorBlock],
      srcOutBlocks: RDD[OutBlock],
      dstInBlocks: RDD[InBlock],
      k: Int): RDD[FactorBlock] = {
    val srcOut = srcOutBlocks.zip(srcFactorBlocks).flatMap { case (srcOutBlock, srcFactors) =>
      val srcFactorMap = new OpenIntObjectHashMap(srcFactors.size)
      srcFactors.foreach { case (srcId, srcFactor) =>
        srcFactorMap.put(srcId, srcFactor)
      }
      srcOutBlock.view.zipWithIndex.map { case (srcIds, dstBlockId) =>
        val outputFactors = ArrayBuffer.empty[(Int, Array[Float])]
        srcIds.foreach { i =>
          outputFactors += ((i, srcFactorMap.get(i).asInstanceOf[Array[Float]]))
        }
        (dstBlockId, outputFactors.toArray)
      }
    }
    val merged = srcOut.reduceByKey(new IdentityPartitioner(dstInBlocks.partitions.size), _ ++ _).values
    dstInBlocks.zip(merged).map { case (InBlock(dstIds, srcPtrs, srcIds, ratings), srcFactors) =>
      val srcFactorMap = new OpenIntObjectHashMap(srcFactors.size)
      srcFactors.foreach { case (srcId, srcFactor) =>
        srcFactorMap.put(srcId, srcFactor)
      }
      val dstFactors = ArrayBuffer.empty[(Int, Array[Float])]
      var j = 0
      while (j < dstIds.length) {
        val dstId = dstIds(j)
        val ls = new LeastSquares(k)
        var i = srcPtrs(j)
        while (i < srcPtrs(j + 1)) {
          val srcId = srcIds(i)
          ls.add(srcFactorMap.get(srcId).asInstanceOf[Array[Float]], ratings(i))
          i += 1
        }
        dstFactors += ((dstId, ls.solve(lambda = 0.1f)))
        j += 1
      }
      dstFactors.toArray
    }
  }
}
