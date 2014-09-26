package als

import java.util

import org.apache.spark.{HashPartitioner, Partitioner}
import org.apache.spark.mllib.recommendation.Rating

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

import com.github.fommil.netlib.F2jBLAS
import com.github.fommil.netlib.LAPACK.{getInstance => lapack}
import org.netlib.util.intW

import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD

class ALS {

  import ALS._

  val k = 10
  var userFactors: RDD[FactorBlock] = _
  var prodFactors: RDD[FactorBlock] = _
  var userAssignments: RDD[AssignmentBlock] = _
  var prodAssignments: RDD[AssignmentBlock] = _
  var userComputations: RDD[ComputationBlock] = _
  var prodComputations: RDD[ComputationBlock] = _

  def run(ratings: RDD[Rating], numBlocks: Int): (RDD[FactorBlock], RDD[FactorBlock]) = {
    val srcPart = new HashPartitioner(numBlocks)
    val dstPart = new HashPartitioner(numBlocks)
    val blocks = ratings.groupBy(x => (srcPart.getPartition(x.user), dstPart.getPartition(x.product))).mapValues(_.toSeq)
    blocks.mapValues(optimizeBlock(_, k))
    null
  }
}

class LeastSquares(val k: Int) {

  val triK = k * (k + 1) / 2
  val ata = new Array[Double](triK)
  val atb = new Array[Double](k)
  val da = new Array[Double](k)
  var n = 0
  val info = new intW(0)
  val upper = "U"

  import LeastSquares._

  def add(a: Array[Float], b: Float): this.type = {
    var i = 0
    while (i < k) {
      da(i) = a(i)
      i += 1
    }
    blas.dspr(upper, k, 1.0, da, 1, ata)
    blas.daxpy(k, b.toDouble, da, 1, atb, 1)
    n += 1
    this
  }

  def merge(other: LeastSquares): this.type = {
    blas.daxpy(ata.size, 1.0, other.ata, 1, ata, 1)
    blas.daxpy(atb.size, 1.0, other.atb, 1, atb, 1)
    n += other.n
    this
  }

  def solve(lambda: Float): Array[Float] = {
    val scaledlambda = lambda * n
    var i = 0
    var j = 2
    while (i < triK) {
      ata(i) += scaledlambda
      i += j
      j += 1
    }
    lapack.dppsv(upper, k, 1, ata, atb, k, info)
    val code = info.`val`
    assert(code == 0, s"lapack.sppsv returned $code.")
    val x = new Array[Float](k)
    i = 0
    while (i < k) {
      x(i) = atb(i).toFloat
      i += 1
    }
    reset()
    x
  }

  def reset(): Unit = {
    util.Arrays.fill(ata, 0.0)
    util.Arrays.fill(atb, 0.0)
    n = 0
  }
}

object LeastSquares {
  val blas = new F2jBLAS
}

class IdentityPartitioner(override val numPartitions: Int) extends Partitioner {
  override def getPartition(key: Any): Int = key.asInstanceOf[Int]
  override def equals(other: Any): Boolean = {
    other match {
      case p: IdentityPartitioner =>
        this.numPartitions == p.numPartitions
      case _ =>
        false
    }
  }
}

object ALS {

  val blas = new F2jBLAS

  // (blockId, factors)
  type FactorBlock = (Int, Map[Int, Array[Float]])
  type AssignmentBlock = (Int, Seq[Assignment])
  type ComputationBlock = (Int, Computation)

  def optimizeBlock(ratings: Seq[Rating], k: Int): Unit = {
    val countByUser = mutable.Map.empty[Int, Int]
    ratings.foreach { r =>
      val c = countByUser.getOrElse(r.user, 0)
      countByUser(r.user) = c + 1
    }
    val groupByProduct = ratings.groupBy(r => r.product).toSeq.sortBy(-_._2.size).toArray
    val numProducts = groupByProduct.size
    val numUsers = countByUser.size
    var cost = 1.0 * numUsers * (k + 1)
    var bestCost = cost
    var bestDirectStart = 0
    var i = 1
    groupByProduct.foreach { case (p, rr) =>
      cost += k * (k + 3) / 2
      rr.foreach { r =>
        countByUser(r.user) -= 1
        if (countByUser(r.user) == 0) {
          cost -= k + 1
        }
      }
      if (cost < bestCost) {
        bestCost = cost
        bestDirectStart = i
      }
      i += 1
    }
    def compress(start: Int, end: Int): (Array[Int], Array[Int], Array[Int], Array[Double]) = {
      val size = end - start
      val products = new Array[Int](size)
      val userCountsBuffer = ArrayBuffer.empty[Int]
      val usersBuffer = ArrayBuffer.empty[Int]
      val ratingsBuffer = ArrayBuffer.empty[Double]
      i = start
      while (i < end) {
        val (product, rr) = groupByProduct(i)
        products(i) = product
        userCountsBuffer += rr.size
        usersBuffer ++= rr.map(_.user)
        ratingsBuffer ++= rr.map(_.rating)
      }
      val userPtrs = userCountsBuffer.scanLeft(0)(_ + _).toArray
      val users = usersBuffer.toArray
      val ratings = ratingsBuffer.toArray
      (products, userPtrs, users, ratings)
    }
    val (matrixProducts, matrixUserPtrs, matrixUsers, matrixRatings) =
      compress(0, bestDirectStart)
    val (directProducts, directUserPtrs, directUsers, directRatings) =
      compress(bestDirectStart, numProducts)
  }


  def makeAssignment(ratings: Seq[Rating], k: Int): (Assignment, Computation) = {
    val countBySrc = mutable.Map.empty[Int, Int]
    ratings.foreach { r =>
      val c = countBySrc.getOrElse(r.user, 0)
      countBySrc(r.user) = c + 1
    }
    val groupByDst = ratings.groupBy(r => r.product).toSeq.sortBy(-_._2.size).toArray
    val numProducts = groupByDst.size
    val numUsers = countBySrc.size
    var cost = 1.0 * numUsers * (k + 1)
    var bestCost = cost
    var bestDirectStart = 0
    var i = 1
    groupByDst.foreach { case (p, rr) =>
      cost += k * (k + 3) / 2
      rr.foreach { r =>
        countBySrc(r.user) -= 1
        if (countBySrc(r.user) == 0) {
          cost -= k + 1
        }
      }
      if (cost < bestCost) {
        bestCost = cost
        bestDirectStart = i
      }
      i += 1
    }
    def compress(start: Int, end: Int): (Array[Int], Array[Int], Array[Int], Array[Double]) = {
      val size = end - start
      val products = new Array[Int](size)
      val userCountsBuffer = ArrayBuffer.empty[Int]
      val usersBuffer = ArrayBuffer.empty[Int]
      val ratingsBuffer = ArrayBuffer.empty[Double]
      i = start
      while (i < end) {
        val (product, rr) = groupByDst(i)
        products(i) = product
        userCountsBuffer += rr.size
        usersBuffer ++= rr.map(_.user)
        ratingsBuffer ++= rr.map(_.rating)
      }
      val userPtrs = userCountsBuffer.scanLeft(0)(_ + _).toArray
      val users = usersBuffer.toArray
      val ratings = ratingsBuffer.toArray
      (products, userPtrs, users, ratings)
    }
    val (matrixProducts, matrixUserPtrs, matrixUsers, matrixRatings) =
      compress(0, bestDirectStart)
    val (directProducts, directUserPtrs, directUsers, directRatings) =
      compress(bestDirectStart, numProducts)
    null
  }

  case class Assignment(
      dstBlockId: Int,
      directSrcIds: Array[Int],
      matrixDstIds: Array[Int],
      matrixSrcPtrs: Array[Int],
      matrixSrcIds: Array[Int],
      matrixRatings: Array[Float])

  case class AssignmentOutput(
     factors: mutable.ListBuffer[(Int, Array[Float])],
     matrices: mutable.Map[Int, LeastSquares]) {

    def merge(other: AssignmentOutput): this.type = {
      factors ++= other.factors
      other.matrices.foreach { case (dstId, ls) =>
        if (!matrices.contains(dstId)) {
          matrices(dstId) = ls
        } else {
          matrices(dstId).merge(ls)
        }
      }
      this
    }
  }

  case class Computation(
      dstIds: Array[Int],
      directSrcPtrs: Array[Int],
      directSrcIds: Array[Int],
      directRatings: Array[Float])

  def computeFactors(
      factors: RDD[FactorBlock],
      assignments: RDD[AssignmentBlock],
      computations: RDD[ComputationBlock],
      k: Int): RDD[FactorBlock] = {
    val triK = k * (k + 1) / 2
    val assignmentOutputs = factors.join(assignments).values.flatMap { case (thisFactors, thisAssignments) =>
       thisAssignments.map { case Assignment(dstBlockId, directSrcIds, matrixDstIds, matrixSrcPtrs, matrixSrcIds, matrixRatings) =>
         val directFactors = directSrcIds.map(i => (i, thisFactors(i)))
         val matrices = mutable.Map.empty[Int, LeastSquares] // TODO: append-only map
         var j = 0
         while (j < matrixDstIds.length) {
           val dstId = matrixDstIds(j)
           var i = matrixSrcPtrs(j)
           val ls = new LeastSquares(k)
           while (i < matrixSrcPtrs(j + 1)) {
             val srcId = matrixSrcIds(i)
             ls.add(thisFactors(srcId), matrixRatings(i))
             i += 1
           }
           matrices += ((dstId, ls))
           j += 1
         }
         (dstBlockId, AssignmentOutput(mutable.ListBuffer.empty ++ directFactors.toSeq, matrices))
       }
    }
    val merged = assignmentOutputs.reduceByKey(computations.partitioner.get, (o1, o2) => o1.merge(o2))
    val outputFactors = computations.join(merged).mapValues { case (Computation(dstIds, directSrcPtrs, directSrcIds, directRatings), AssignmentOutput(srcFactors, matrices)) =>
      val srcFactorMap = srcFactors.toMap
      val dstFactors = mutable.Map.empty[Int, Array[Float]]
      var j = 0
      while (j < dstIds.length) {
        val dstId = dstIds(j)
        val ls = matrices.getOrElse(dstId, new LeastSquares(k))
        var i = directSrcPtrs(j)
        while (i < directSrcPtrs(j + 1)) {
          val srcId = directSrcIds(i)
          ls.add(srcFactorMap(srcId), directRatings(i))
          i += 1
        }
        dstFactors += ((dstId, ls.solve(lambda = 0.1f)))
        j += 1
      }
      dstFactors.toMap
    }
    outputFactors
  }
}
