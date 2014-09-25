package org.apache.spark.ml

import java.nio.{ByteBuffer, IntBuffer}
import java.{util => javaUtil}

import als.{GridPartitioner, IdentityPartitioner, LeastSquares}
import com.github.fommil.netlib.BLAS.{getInstance => blas}
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD
import org.apache.spark.{Logging, HashPartitioner, Partitioner}
import org.apache.spark.util.collection.{OpenHashMap, OpenHashSet, SortDataFormat}
import org.apache.spark.ml.util.Sorter

import scala.collection.mutable.ArrayBuffer

class SimpleALS extends Logging {

  import org.apache.spark.ml.SimpleALS._

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
    blockRatings.unpersist()
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

  class BufferedRatingBlock extends Serializable {

    private val srcIds: ArrayBuffer[Int] = ArrayBuffer.empty
    private val dstIds: ArrayBuffer[Int] = ArrayBuffer.empty
    private val ratings: ArrayBuffer[Float] = ArrayBuffer.empty

    def add(srcId: Int, dstId: Int, rating: Float): this.type = {
      srcIds += srcId
      dstIds += dstId
      ratings += rating
      this
    }

    def merge(other: BufferedRatingBlock): this.type = {
      srcIds ++= other.srcIds
      dstIds ++= other.dstIds
      ratings ++= other.ratings
      this
    }

    def merge(other: RatingBlock): this.type = {
      srcIds ++= other.srcIds
      dstIds ++= other.dstIds
      ratings ++= other.localRatings
      this
    }

    def size: Int = srcIds.size

    def toRatingBlock: RatingBlock = {
      RatingBlock(srcIds.toArray, dstIds.toArray, ratings.toArray)
    }
  }

  def blockifyRatings(ratings: RDD[(Int, Int, Float)], srcPart: Partitioner, dstPart: Partitioner): RDD[((Int, Int), RatingBlock)] = {
    val gridPart = new GridPartitioner(Array(srcPart, dstPart))
    ratings.mapPartitions { iter =>
      val blocks = Array.fill(gridPart.numPartitions)(new BufferedRatingBlock)
      iter.flatMap { case (srcId, dstId, rating) =>
        val idx = gridPart.getPartition((srcId, dstId))
        val block = blocks(idx)
        block.add(srcId, dstId, rating)
        if (block.size >= 2048) { // 2048 * (3 * 4) = 24k
          blocks(idx) = new BufferedRatingBlock
          val Array(srcBlockId, dstBlockId) = gridPart.indices(idx)
          Iterator.single(((srcBlockId, dstBlockId), block.toRatingBlock))
        } else {
          Iterator.empty
        }
      } ++ {
        blocks.view.zipWithIndex.filter(_._1.size > 0).map { case (block, idx) =>
          val Array(srcBlockId, dstBlockId) = gridPart.indices(idx)
          ((srcBlockId, dstBlockId), block.toRatingBlock)
        }
      }
    }.groupByKey().mapValues { iter =>
      val buffered = new BufferedRatingBlock
      iter.foreach(buffered.merge)
      buffered.toRatingBlock
    }.setName("blockRatings")
  }

  class UncompressedBlockBuilder {

    val srcIds = ArrayBuffer.empty[Int]
    val dstBlockIds = ArrayBuffer.empty[Short]
    val dstLocalIndices = ArrayBuffer.empty[Int]
    val ratings = ArrayBuffer.empty[Float]

    def add(theDstBlockId: Short, theSrcIds: Array[Int], theDstLocalIndices: Array[Int], theRatings: Array[Float]): this.type = {
      val sz = theSrcIds.size
      require(theDstLocalIndices.size == sz)
      require(theRatings.size == sz)
      srcIds ++= theSrcIds
      val shortDstBlockId = theDstBlockId.toShort
      var j = 0
      while (j < sz) {
        dstBlockIds += shortDstBlockId
        j += 1
      }
      dstLocalIndices ++= theDstLocalIndices
      ratings ++= theRatings
      this
    }

    def build(): UncompressedBlock = {
      new UncompressedBlock(srcIds.toArray, dstBlockIds.toArray, dstLocalIndices.toArray, ratings.toArray)
    }
  }

  class UncompressedBlock(
      val srcIds: Array[Int],
      val dstBlockIds: Array[Short],
      val dstLocalIndices: Array[Int],
      val ratings: Array[Float]) {

    def size: Int = srcIds.size

    def compress(): InBlock = {
      val sz = size
      assert(sz > 0)
      sort()
      var preSrcId = srcIds(0)
      val uniqueSrcIds = ArrayBuffer(preSrcId)
      val dstCounts = ArrayBuffer(1)
      var i = 1
      var j = 0
      while (i < sz) {
        val srcId = srcIds(i)
        if (srcId != preSrcId) {
          uniqueSrcIds += srcId
          dstCounts += 0
          preSrcId = srcId
          j += 1
        }
        dstCounts(j) += 1
        i += 1
      }
      val dstPtrs = dstCounts.scanLeft(0)(_ + _).toArray
      InBlock(uniqueSrcIds.toArray, dstPtrs, dstBlockIds.toArray, dstLocalIndices.toArray, ratings.toArray)
    }

    private def indexSort(): Unit = {
      val sz = size

      val indices = new Array[Int](sz)
      var i = 0
      while (i < sz) {
        indices(i) = i
        i += 1
      }
      def ord(i0: Int, i1: Int): Boolean = {
        srcIds(i0) < srcIds(i1)
      }
      val sortedIndices = indices.sortWith(ord)

      val sortedSrcIds = new Array[Int](sz)
      val sortedDstBlockIds = new Array[Short](sz)
      val sortedDstLocalIndices = new Array[Int](sz)
      val sortedRatings = new Array[Float](sz)

      i = 0
      while (i < sz) {
        val idx = sortedIndices(i)
        sortedSrcIds(i) = srcIds(idx)
        sortedDstBlockIds(i) = dstBlockIds(idx)
        sortedDstLocalIndices(i) = dstLocalIndices(idx)
        sortedRatings(i) = ratings(idx)
        i += 1
      }

      System.arraycopy(sortedSrcIds, 0, srcIds, 0, sz)
      System.arraycopy(sortedDstBlockIds, 0, dstBlockIds, 0, sz)
      System.arraycopy(sortedDstLocalIndices, 0, dstLocalIndices, 0, sz)
      System.arraycopy(sortedRatings, 0, ratings, 0, sz)
    }

    private def timSort(): Unit = {
      val sorter = new Sorter(new LocalRatingBlockSort)
      sorter.sort(this, 0, size, Ordering[Int])
    }

    private def scalaSort(): Unit = {
      val sz = size
      val sorted = (0 until sz).map { i =>
        (srcIds(i), dstBlockIds(i), dstLocalIndices(i), ratings(i))
      }.sortBy(_._1)
      var i = 0
      sorted.foreach { case (srcId, dstBlockId, dstLocalIndex, rating) =>
        srcIds(i) = srcId
        dstBlockIds(i) = dstBlockId
        dstLocalIndices(i) = dstLocalIndex
        ratings(i) = rating
        i += 1
      }
    }

    private def sort(): Unit = {
      val sz = size
      println("size: " + sz)
      val start = System.nanoTime()
      indexSort()
      // timSort()
      // scalaSort()
      println("sort uncompressed time: " + (System.nanoTime() - start) / 1e9)
    }

    override def toString: String = {
      "srcIds: " + srcIds.toSeq + "\n" +
          "dstBlockIds: " + dstBlockIds.toSeq + "\n" +
          "dstLocalIndices: " + dstLocalIndices.toSeq + "\n" +
          "ratings: " + ratings.toSeq
    }
  }

  class LocalRatingBlockSort extends SortDataFormat[Int, UncompressedBlock] {

    override protected def getKey(data: UncompressedBlock, pos: Int): Int = {
      data.srcIds(pos)
    }

    private def swapElements[@specialized(Int, Float, Short) T](data: Array[T], pos0: Int, pos1: Int): Unit = {
      val tmp = data(pos0)
      data(pos0) = data(pos1)
      data(pos1) = tmp
    }

    override protected def swap(data: UncompressedBlock, pos0: Int, pos1: Int): Unit = {
      swapElements(data.srcIds, pos0, pos1)
      swapElements(data.dstBlockIds, pos0, pos1)
      swapElements(data.dstLocalIndices, pos0, pos1)
      swapElements(data.ratings, pos0, pos1)
    }

    override protected def copyRange(
        src: UncompressedBlock,
        srcPos: Int,
        dst: UncompressedBlock,
        dstPos: Int,
        length: Int): Unit = {
      System.arraycopy(src.srcIds, srcPos, dst.srcIds, dstPos, length)
      System.arraycopy(src.dstBlockIds, srcPos, dst.dstBlockIds, dstPos, length)
      System.arraycopy(src.dstLocalIndices, srcPos, dst.dstLocalIndices, dstPos, length)
      System.arraycopy(src.ratings, srcPos, dst.ratings, dstPos, length)
    }

    override protected def allocate(length: Int): UncompressedBlock = {
      new UncompressedBlock(new Array[Int](length), new Array[Short](length), new Array[Int](length), new Array[Float](length))
    }

    override protected def copyElement(
        src: UncompressedBlock,
        srcPos: Int,
        dst: UncompressedBlock,
        dstPos: Int): Unit = {
      dst.srcIds(dstPos) = src.srcIds(srcPos)
      dst.dstBlockIds(dstPos) = src.dstBlockIds(srcPos)
      dst.dstLocalIndices(dstPos) = src.dstLocalIndices(srcPos)
      dst.ratings(dstPos) = src.ratings(srcPos)
    }
  }

  def makeBlocks(
      prefix: String,
      ratingBlocks: RDD[((Int, Int), RatingBlock)],
      srcPart: Partitioner,
      dstPart: Partitioner): (RDD[(Int, InBlock)], RDD[OutBlock]) = {
    val inBlocks = ratingBlocks.map { case ((srcBlockId, dstBlockId), RatingBlock(srcIds, dstIds, ratings)) =>
      // faster version of
      var start = System.nanoTime()
      // val slowDstIdToLocalIndex = dstIds.toSet.toSeq.sorted.zipWithIndex.toMap
      // println("slow sort time: " + (System.nanoTime() - start) / 1e9)
      // start = System.nanoTime()
      val dstIdSet = new OpenHashSet[Int](1 << 20)
      dstIds.foreach(dstIdSet.add)
      val sortedDstIds = dstIdSet.iterator.toArray.sorted
      val dstIdToLocalIndex = new OpenHashMap[Int, Int](sortedDstIds.size)
      var i = 0
      while (i < sortedDstIds.size) {
        dstIdToLocalIndex.update(sortedDstIds(i), i)
        i += 1
      }
      println("fast sort time: " + (System.nanoTime() - start) / 1e9)
      val dstLocalIndices = dstIds.map(dstIdToLocalIndex.apply)
      (srcBlockId, (dstBlockId, srcIds, dstLocalIndices, ratings))
    }.groupByKey(new IdentityPartitioner(srcPart.numPartitions))
        .mapValues { iter =>
      val uncompressedBlockBuilder = new UncompressedBlockBuilder
      iter.foreach { case (dstBlockId, srcIds, dstLocalIndices, ratings) =>
        uncompressedBlockBuilder.add(dstBlockId.toShort, srcIds, dstLocalIndices, ratings)
      }
      uncompressedBlockBuilder.build().compress()
    }.setName(prefix + "InBlocks").cache()
    val outBlocks = inBlocks.mapValues { case InBlock(srcIds, dstPtrs, dstBlockIds, _, _) =>
      val activeIds = Array.fill(dstPart.numPartitions)(ArrayBuffer.empty[Int])
      var i = 0
      val seen = new Array[Boolean](dstPart.numPartitions)
      while (i < srcIds.size) {
        var j = dstPtrs(i)
        javaUtil.Arrays.fill(seen, false)
        while (j < dstPtrs(i + 1)) {
          val dstBlockId = dstBlockIds(j)
          if (!seen(dstBlockId)) {
            activeIds(dstBlockId) += i
            seen(dstBlockId) = true
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
      val ls = new LeastSquares(k)
      while (j < dstIds.size) {
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
