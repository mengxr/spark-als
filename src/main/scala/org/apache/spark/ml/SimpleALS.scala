package org.apache.spark.ml

import java.{util => javaUtil}

import scala.collection.mutable.ArrayBuilder

import com.github.fommil.netlib.BLAS.{getInstance => blas}

import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD
import org.apache.spark.util.random.XORShiftRandom
import org.apache.spark.{HashPartitioner, Partitioner}
import org.apache.spark.util.collection.{Sorter, SortDataFormat, OpenHashMap, OpenHashSet}
import org.apache.spark.ml.util._

class Rating(val user: Int, val product: Int, val rating: Float) extends Serializable

object Rating {
  def parseRating(str: String, delimiter: String): Rating = {
    val fields = str.split(delimiter)
    assert(fields.size >= 3)
    val user = java.lang.Integer.parseInt(fields(0))
    val product = java.lang.Integer.parseInt(fields(1))
    val rating = java.lang.Float.parseFloat(fields(2))
    new Rating(user, product, rating)
  }
}

class SimpleALS extends Serializable {

  import org.apache.spark.ml.SimpleALS._

  def run(
      ratings: RDD[Rating],
      k: Int = 10,
      numBlocks: Int = 10,
      numIterations: Int = 10,
      lambda: Double = 1.0,
      implicitPrefs: Boolean = false,
      alpha: Double = 1.0): (RDD[(Int, Array[Float])], RDD[(Int, Array[Float])]) = {
    val userPart = new HashPartitioner(numBlocks)
    val prodPart = new HashPartitioner(numBlocks)
    val userLocalIndexEncoder = new LocalIndexEncoder(userPart.numPartitions)
    val prodLocalIndexEncoder = new LocalIndexEncoder(prodPart.numPartitions)
    val blockRatings = blockifyRatings(ratings, userPart, prodPart).cache()
    val (userInBlocks, userOutBlocks) = makeBlocks("user", blockRatings, userPart, prodPart)
    // materialize blockRatings and user blocks
    userOutBlocks.count()
    val swappedBlockRatings = blockRatings.map {
      case ((userBlockId, prodBlockId), RatingBlock(userIds, prodIds, localRatings)) =>
        ((prodBlockId, userBlockId), RatingBlock(prodIds, userIds, localRatings))
    }
    val (prodInBlocks, prodOutBlocks) = makeBlocks("prod", swappedBlockRatings, prodPart, userPart)
    // materialize prod blocks
    prodOutBlocks.count()
    var userFactors = initialize(userInBlocks, k)
    var prodFactors = initialize(prodInBlocks, k)
    if (implicitPrefs) {
      for (iter <- 1 to numIterations) {
        userFactors.setName(s"userFactors-$iter").persist()
        val YtY = Some(computeYtY(userFactors, k))
        val previousProdFactors = prodFactors
        prodFactors = computeFactors(userFactors, userOutBlocks, prodInBlocks, k, lambda, userLocalIndexEncoder, implicitPrefs, alpha, YtY)
        previousProdFactors.unpersist()
        prodFactors.setName(s"prodFactors-$iter").persist()
        val XtX = Some(computeYtY(prodFactors, k))
        val previousUserFactors = userFactors
        userFactors = computeFactors(prodFactors, prodOutBlocks, userInBlocks, k, lambda, prodLocalIndexEncoder, implicitPrefs, alpha, XtX)
        previousUserFactors.unpersist()
      }
    } else {
      for (iter <- 0 until numIterations) {
        prodFactors = computeFactors(userFactors, userOutBlocks, prodInBlocks, k, lambda, userLocalIndexEncoder)
        userFactors = computeFactors(prodFactors, prodOutBlocks, userInBlocks, k, lambda, prodLocalIndexEncoder)
      }
    }
    val userIdAndFactors = userInBlocks.mapValues(_.srcIds).join(userFactors).values.setName("userFactors").cache()
    userIdAndFactors.count()
    prodFactors.unpersist()
    val prodIdAndFactors = prodInBlocks.mapValues(_.srcIds).join(prodFactors).values.setName("prodFactors").cache()
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

private object SimpleALS {

  /**
   * Factor block that stores factors (Array[Float]) in an Array.
   */
  private type FactorBlock = Array[Array[Float]]

  /**
   * Out block that stores, for each dst block, which src factors to send.
   * For example, outBlock(0) contains the indices of the src factors to send to dst block 0.
   */
  private type OutBlock = Array[Array[Int]]

  /**
   * In block for computing src factors.
   *
   * For each src id, it stores its associated dst block ids and local indices, and ratings.
   * So given the dst factors, it is easy to compute src factors one by one.
   * We use compressed sparse column (CSC) format.
   *
   * @param srcIds src ids (ordered)
   * @param dstPtrs dst pointers. Elements in range [dstPtrs(i), dstPtrs(i+1)) of dst indices and
   *                ratings are associated with srcIds(i).
   * @param dstEncodedIndices encoded dst indices
   * @param ratings ratings
   */
  private case class InBlock(
      srcIds: Array[Int],
      dstPtrs: Array[Int],
      dstEncodedIndices: Array[Int],
      ratings: Array[Float])

  /**
   * Initializes factors randomly given the in blocks.
   *
   * @param inBlocks in blocks
   * @param k rank
   * @return initialized factor blocks
   */
  private def initialize(inBlocks: RDD[(Int, InBlock)], k: Int): RDD[(Int, FactorBlock)] = {
    inBlocks.map { case (srcBlockId, inBlock) =>
      val random = new XORShiftRandom(srcBlockId)
      val factors = Array.fill(inBlock.srcIds.size) {
        val factor = Array.fill(k)(random.nextGaussian().toFloat)
        val nrm = blas.snrm2(k, factor, 1)
        blas.sscal(k, 1.0f / nrm, factor, 1)
        factor
      }
      (srcBlockId, factors)
    }
  }

  /**
   * A rating block that contains src ids, dst ids, and ratings.
   *
   * @param srcIds src ids
   * @param dstIds dst ids
   * @param ratings ratings
   */
  private case class RatingBlock(srcIds: Array[Int], dstIds: Array[Int], ratings: Array[Float])

  /**
   * Builder for [[RatingBlock]].
   */
  private class RatingBlockBuilder extends Serializable {

    private val srcIds = ArrayBuilder.make[Int]
    private val dstIds = ArrayBuilder.make[Int]
    private val ratings = ArrayBuilder.make[Float]
    var size = 0

    def add(r: Rating): this.type = {
      size += 1
      srcIds += r.user
      dstIds += r.product
      ratings += r.rating
      this
    }

    def merge(other: RatingBlock): this.type = {
      size += other.srcIds.size
      srcIds ++= other.srcIds
      dstIds ++= other.dstIds
      ratings ++= other.ratings
      this
    }

    def toRatingBlock: RatingBlock = {
      RatingBlock(srcIds.result(), dstIds.result(), ratings.result())
    }
  }

  private def blockifyRatings(
      ratings: RDD[Rating],
      srcPart: Partitioner,
      dstPart: Partitioner): RDD[((Int, Int), RatingBlock)] = {
    val gridPart = new GridPartitioner(Array(srcPart, dstPart))
    ratings.mapPartitions { iter =>
      val blocks = Array.fill(gridPart.numPartitions)(new RatingBlockBuilder)
      iter.flatMap { r =>
        val idx = gridPart.getPartition((r.user, r.product))
        val block = blocks(idx)
        block.add(r)
        if (block.size >= 2048) { // 2048 * (3 * 4) = 24k
          blocks(idx) = new RatingBlockBuilder
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
      val buffered = new RatingBlockBuilder
      iter.foreach(buffered.merge)
      buffered.toRatingBlock
    }.setName("blockRatings")
  }

  private class UncompressedBlockBuilder(encoder: LocalIndexEncoder) {

    val srcIds = ArrayBuilder.make[Int]
    val dstEncodedIndices = ArrayBuilder.make[Int]
    val ratings = ArrayBuilder.make[Float]

    def add(
        theDstBlockId: Int,
        theSrcIds: Array[Int],
        theDstLocalIndices: Array[Int],
        theRatings: Array[Float]): this.type = {
      val sz = theSrcIds.size
      require(theDstLocalIndices.size == sz)
      require(theRatings.size == sz)
      srcIds ++= theSrcIds
      ratings ++= theRatings
      var j = 0
      while (j < sz) {
        dstEncodedIndices += encoder.encode(theDstBlockId, theDstLocalIndices(j))
        j += 1
      }
      this
    }

    def build(): UncompressedBlock = {
      new UncompressedBlock(srcIds.result(), dstEncodedIndices.result(), ratings.result())
    }
  }

  private class UncompressedBlock(
      val srcIds: Array[Int],
      val dstEncodedIndices: Array[Int],
      val ratings: Array[Float]) {

    def size: Int = srcIds.size

    def compress(): InBlock = {
      val sz = size
      assert(sz > 0)
      sort()
      val uniqueSrcIdsBuilder = ArrayBuilder.make[Int]
      val dstCountsBuilder = ArrayBuilder.make[Int]
      var preSrcId = srcIds(0)
      uniqueSrcIdsBuilder += preSrcId
      var curCount = 1
      var i = 1
      var j = 0
      while (i < sz) {
        val srcId = srcIds(i)
        if (srcId != preSrcId) {
          uniqueSrcIdsBuilder += srcId
          dstCountsBuilder += curCount
          preSrcId = srcId
          j += 1
          curCount = 0
        }
        curCount += 1
        i += 1
      }
      dstCountsBuilder += curCount
      val uniqueSrcIds = uniqueSrcIdsBuilder.result()
      val numUniqueSrdIds = uniqueSrcIds.size
      val dstCounts = dstCountsBuilder.result()
      val dstPtrs = new Array[Int](numUniqueSrdIds + 1)
      var sum = 0
      i = 0
      while (i < numUniqueSrdIds) {
        sum += dstCounts(i)
        i += 1
        dstPtrs(i) = sum
      }
      InBlock(uniqueSrcIds, dstPtrs, dstEncodedIndices, ratings)
    }

    private def timSort(): Unit = {
      val sorter = new Sorter(new UncompressedBlockSort)
      sorter.sort(this, 0, size, Ordering[IntWrapper])
    }

    private def sort(): Unit = {
      val sz = size
      println("size: " + sz)
      val start = System.nanoTime()
      timSort()
      println("sort uncompressed time: " + (System.nanoTime() - start) / 1e9)
    }
  }

  private class IntWrapper(var key: Int = 0) extends Ordered[IntWrapper] {
    override def compare(that: IntWrapper): Int = {
      key.compare(that.key)
    }
  }

  private class UncompressedBlockSort extends SortDataFormat[IntWrapper, UncompressedBlock] {

    override def newKey(): IntWrapper = new IntWrapper()

    override def getKey(
        data: UncompressedBlock,
        pos: Int,
        reuse: IntWrapper): IntWrapper = {
      if (reuse == null) {
        new IntWrapper(data.srcIds(pos))
      } else {
        reuse.key = data.srcIds(pos)
        reuse
      }
    }

    override def getKey(
        data: UncompressedBlock,
        pos: Int): IntWrapper = {
      getKey(data, pos, null)
    }

    private def swapElements[@specialized(Int, Float) T](
        data: Array[T],
        pos0: Int,
        pos1: Int): Unit = {
      val tmp = data(pos0)
      data(pos0) = data(pos1)
      data(pos1) = tmp
    }

    override def swap(data: UncompressedBlock, pos0: Int, pos1: Int): Unit = {
      swapElements(data.srcIds, pos0, pos1)
      swapElements(data.dstEncodedIndices, pos0, pos1)
      swapElements(data.ratings, pos0, pos1)
    }

    override def copyRange(
        src: UncompressedBlock,
        srcPos: Int,
        dst: UncompressedBlock,
        dstPos: Int,
        length: Int): Unit = {
      System.arraycopy(src.srcIds, srcPos, dst.srcIds, dstPos, length)
      System.arraycopy(src.dstEncodedIndices, srcPos, dst.dstEncodedIndices, dstPos, length)
      System.arraycopy(src.ratings, srcPos, dst.ratings, dstPos, length)
    }

    override def allocate(length: Int): UncompressedBlock = {
      new UncompressedBlock(
        new Array[Int](length), new Array[Int](length), new Array[Float](length))
    }

    override def copyElement(
        src: UncompressedBlock,
        srcPos: Int,
        dst: UncompressedBlock,
        dstPos: Int): Unit = {
      dst.srcIds(dstPos) = src.srcIds(srcPos)
      dst.dstEncodedIndices(dstPos) = src.dstEncodedIndices(srcPos)
      dst.ratings(dstPos) = src.ratings(srcPos)
    }
  }

  private def makeBlocks(
      prefix: String,
      ratingBlocks: RDD[((Int, Int), RatingBlock)],
      srcPart: Partitioner,
      dstPart: Partitioner): (RDD[(Int, InBlock)], RDD[(Int, OutBlock)]) = {
    val inBlocks = ratingBlocks.map {
      case ((srcBlockId, dstBlockId), RatingBlock(srcIds, dstIds, ratings)) =>
        // faster version of
        // val slowDstIdToLocalIndex = dstIds.toSet.toSeq.sorted.zipWithIndex.toMap
        val start = System.nanoTime()
        val dstIdSet = new OpenHashSet[Int](1 << 20)
        dstIds.foreach(dstIdSet.add)
        val sortedDstIds = new Array[Int](dstIdSet.size)
        var i = 0
        var pos = dstIdSet.nextPos(0)
        while (pos != -1) {
          sortedDstIds(i) = dstIdSet.getValue(pos)
          pos = dstIdSet.nextPos(pos + 1)
          i += 1
        }
        assert(i == dstIdSet.size)
        javaUtil.Arrays.sort(sortedDstIds)
        val dstIdToLocalIndex = new OpenHashMap[Int, Int](sortedDstIds.size)
        i = 0
        while (i < sortedDstIds.size) {
          dstIdToLocalIndex.update(sortedDstIds(i), i)
          i += 1
        }
        println("convert to local indices time: " + (System.nanoTime() - start) / 1e9)
        val dstLocalIndices = dstIds.map(dstIdToLocalIndex.apply)
        (srcBlockId, (dstBlockId, srcIds, dstLocalIndices, ratings))
    }.groupByKey(new IdentityPartitioner(srcPart.numPartitions))
        .mapValues { iter =>
      val uncompressedBlockBuilder =
        new UncompressedBlockBuilder(new LocalIndexEncoder(dstPart.numPartitions))
      iter.foreach { case (dstBlockId, srcIds, dstLocalIndices, ratings) =>
        uncompressedBlockBuilder.add(dstBlockId, srcIds, dstLocalIndices, ratings)
      }
      uncompressedBlockBuilder.build().compress()
    }.setName(prefix + "InBlocks").cache()
    val outBlocks = inBlocks.mapValues { case InBlock(srcIds, dstPtrs, dstEncodedIndices, _) =>
      val encoder = new LocalIndexEncoder(dstPart.numPartitions)
      val activeIds = Array.fill(dstPart.numPartitions)(ArrayBuilder.make[Int])
      var i = 0
      val seen = new Array[Boolean](dstPart.numPartitions)
      while (i < srcIds.size) {
        var j = dstPtrs(i)
        javaUtil.Arrays.fill(seen, false)
        while (j < dstPtrs(i + 1)) {
          val dstBlockId = encoder.blockId(dstEncodedIndices(j))
          if (!seen(dstBlockId)) {
            activeIds(dstBlockId) += i // add the local index
            seen(dstBlockId) = true
          }
          j += 1
        }
        i += 1
      }
      activeIds.map { x =>
        x.result()
      }
    }.setName(prefix + "OutBlocks").cache()
    (inBlocks, outBlocks)
  }

  private def computeFactors(
      srcFactorBlocks: RDD[(Int, FactorBlock)],
      srcOutBlocks: RDD[(Int, OutBlock)],
      dstInBlocks: RDD[(Int, InBlock)],
      k: Int,
      lambda: Double,
      srcEncoder: LocalIndexEncoder,
      implicitPrefs: Boolean = false,
      alpha: Double = 1.0,
      YtY: Option[NormalEquation] = None): RDD[(Int, FactorBlock)] = {
    val srcOut = srcOutBlocks.join(srcFactorBlocks).flatMap {
      case (srcBlockId, (srcOutBlock, srcFactors)) =>
        srcOutBlock.view.zipWithIndex.map { case (activeIndices, dstBlockId) =>
          (dstBlockId, (srcBlockId, activeIndices.map(idx => srcFactors(idx))))
        }
    }
    val merged = srcOut.groupByKey(new IdentityPartitioner(dstInBlocks.partitions.size))
    dstInBlocks.join(merged).mapValues {
      case (InBlock(dstIds, srcPtrs, srcEncodedIndices, ratings), srcFactors) =>
        val sortedSrcFactors = srcFactors.toSeq.sortBy(_._1).map(_._2).toArray
        val dstFactors = new Array[Array[Float]](dstIds.size)
        var j = 0
        val ls = new NormalEquation(k)
        val solver = new CholeskySolver(k)
        while (j < dstIds.size) {
          ls.reset()
          if (implicitPrefs) {
            ls.merge(YtY.get)
          }
          var i = srcPtrs(j)
          while (i < srcPtrs(j + 1)) {
            val encoded = srcEncodedIndices(i)
            val blockId = srcEncoder.blockId(encoded)
            val localIndex = srcEncoder.localIndex(encoded)
            val srcFactor = sortedSrcFactors(blockId)(localIndex)
            val rating = ratings(i)
            if (implicitPrefs) {
              ls.addImplicit(srcFactor, rating, alpha)
            } else {
              ls.add(srcFactor, rating)
            }
            i += 1
          }
          dstFactors(j) = solver.solve(ls, lambda)
          j += 1
        }
        dstFactors
    }
  }

  private def computeYtY(factorBlocks: RDD[(Int, FactorBlock)], k: Int): NormalEquation = {
    factorBlocks.values.aggregate(new NormalEquation(k))(
      seqOp = (ne, factors) => {
        factors.foreach(ne.add(_, 0.0f))
        ne
      },
      combOp = (ne1, ne2) => ne1.merge(ne2))
  }

  /**
   * Encoder for storing (blockId, localIndex) into a single integer.
   *
   * We use the leading bits (including the sign bit) to store the block id and the rest to store
   * the local index. This is based on the assumption that users/products are approximately evenly
   * partitioned. With this assumption, we should be able to encode two billion distinct values.
   *
   * @param numBlocks number of blocks
   */
  private class LocalIndexEncoder(numBlocks: Int) extends Serializable {

    require(numBlocks > 0, s"numBlocks must be positive but found $numBlocks.")

    private[this] final val numLocalIndexBits =
      math.min(java.lang.Integer.numberOfLeadingZeros(numBlocks - 1), 31)
    private[this] final val localIndexMask = (1 << numLocalIndexBits) - 1

    def encode(blockId: Int, localIndex: Int): Int = {
      require(blockId < numBlocks)
      require((localIndex & ~localIndexMask) == 0)
      (blockId << numLocalIndexBits) | localIndex
    }

    @inline
    def blockId(encoded: Int): Int = {
      encoded >>> numLocalIndexBits
    }

    @inline
    def localIndex(encoded: Int): Int = {
      encoded & localIndexMask
    }
  }
}
