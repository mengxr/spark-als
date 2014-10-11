package org.apache.spark.ml

import java.{util => javaUtil}

import com.github.fommil.netlib.F2jBLAS
import com.github.fommil.netlib.LAPACK.{getInstance => lapack}
import org.netlib.util.intW

class CholeskySolver(val k: Int) {

  val upper = "U"
  val info = new intW(0)

  def solve(ne: NormalEquation, lambda: Double): Array[Float] = {
    val scaledlambda = lambda * ne.n
    var i = 0
    var j = 2
    while (i < ne.triK) {
      ne.ata(i) += scaledlambda
      i += j
      j += 1
    }
    lapack.dppsv(upper, k, 1, ne.ata, ne.atb, k, info)
    val code = info.`val`
    assert(code == 0, s"lapack.sppsv returned $code.")
    val x = new Array[Float](k)
    i = 0
    while (i < k) {
      x(i) = ne.atb(i).toFloat
      i += 1
    }
    x
  }
}

class NormalEquation(val k: Int) extends Serializable {

  val triK = k * (k + 1) / 2
  val ata = new Array[Double](triK)
  val atb = new Array[Double](k)
  val da = new Array[Double](k)
  var n = 0
  val upper = "U"

  import NormalEquation._

  private def copyToDouble(a: Array[Float]): Unit = {
    var i = 0
    while (i < k) {
      da(i) = a(i)
      i += 1
    }
  }

  def add(a: Array[Float], b: Float): this.type = {
    copyToDouble(a)
    blas.dspr(upper, k, 1.0, da, 1, ata)
    blas.daxpy(k, b.toDouble, da, 1, atb, 1)
    n += 1
    this
  }

  def addImplicit(a: Array[Float], b: Float, alpha: Double): this.type = {
    val confidence = 1.0 + alpha * math.abs(b)
    copyToDouble(a)
    blas.dspr(upper, k, confidence - 1.0, da, 1, ata)
    if (b > 0) {
      blas.daxpy(k, confidence, da, 1, atb, 1)
    }
    this
  }

  def merge(other: NormalEquation): this.type = {
    blas.daxpy(ata.size, 1.0, other.ata, 1, ata, 1)
    blas.daxpy(atb.size, 1.0, other.atb, 1, atb, 1)
    n += other.n
    this
  }

  def reset(): Unit = {
    javaUtil.Arrays.fill(ata, 0.0)
    javaUtil.Arrays.fill(atb, 0.0)
    n = 0
  }
}

object NormalEquation {
  val blas = new F2jBLAS
}
