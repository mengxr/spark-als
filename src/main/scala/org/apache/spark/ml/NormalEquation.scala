package org.apache.spark.ml

import java.{util => javaUtil}

import com.github.fommil.netlib.F2jBLAS
import com.github.fommil.netlib.LAPACK.{getInstance => lapack}
import org.netlib.util.intW

class NormalEquation(val k: Int) {

  val triK = k * (k + 1) / 2
  val ata = new Array[Double](triK)
  val atb = new Array[Double](k)
  val da = new Array[Double](k)
  var n = 0
  val info = new intW(0)
  val upper = "U"

  import NormalEquation._

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

  def merge(other: NormalEquation): this.type = {
    blas.daxpy(ata.size, 1.0, other.ata, 1, ata, 1)
    blas.daxpy(atb.size, 1.0, other.atb, 1, atb, 1)
    n += other.n
    this
  }

  def solve(lambda: Double): Array[Float] = {
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
    javaUtil.Arrays.fill(ata, 0.0)
    javaUtil.Arrays.fill(atb, 0.0)
    n = 0
  }
}

object NormalEquation {
  val blas = new F2jBLAS
}
