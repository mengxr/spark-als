package als

import org.apache.spark.Partitioner

import scala.collection.mutable

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
}

object ALS {

  // (blockId, factors)
  type FactorBlock = (Int, Map[Int, Array[Float]])
  type AssignmentBlock = (Int, Seq[Assignment])
  type ComputationBlock = (Int, Computation)

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

  case class Assignment(
      dstBlockId: Int,
      directSrcIds: Array[Int],
      matrixDstIds: Array[Int],
      matrixSrcPtrs: Array[Int],
      matrixSrcIds: Array[Int],
      matrixRatings: Array[Float])

  case class LeastSquares(
      ata: Array[Float],
      atb: Array[Float]) {

    def this(k: Int) = this(new Array[Float](k * (k + 1) / 2), new Array[Float](k))

    def add(a: Array[Float], b: Float): this.type = {
      null
      this
    }

    def merge(other: LeastSquares): this.type = {
      null
      this
    }

    def solve(): Array[Float] = {
      null
    }
  }

  case class AssignmentOutput(
     factors: Seq[(Int, Array[Float])],
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
         (dstBlockId, AssignmentOutput(directFactors, matrices))
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
        dstFactors += ((dstId, ls.solve()))
        j += 1
      }
      dstFactors.toMap
    }
    outputFactors
  }
}
