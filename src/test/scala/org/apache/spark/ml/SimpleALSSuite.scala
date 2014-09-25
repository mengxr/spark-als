package org.apache.spark.ml

import com.github.fommil.netlib.BLAS.{getInstance => blas}
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.recommendation.{ALS => MLlibALS}
import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.{BeforeAndAfterAll, FunSuite}

class SimpleALSSuite extends FunSuite with BeforeAndAfterAll {

  var sc: SparkContext = _
  override def beforeAll(): Unit = {
    super.beforeAll()
    val conf = new SparkConf().setMaster("local").setAppName("SimpleALSSuite")
    sc = new SparkContext(conf)
  }

  override def afterAll(): Unit = {
    if (sc != null) {
      sc.stop()
    }
    super.afterAll()
  }

  test("SimpleALS") {
    val ratings = sc.textFile("/Users/meng/share/data/movielens/ml-1m/ratings.dat", 1)
        .map(_.split("::"))
      .flatMap { case Array(u, p, r, t) =>
      (0 until 5).map { i =>
        (u.toInt + i, p.toInt, r.toFloat)
      }
    }
    val Array(training, test) = ratings.randomSplit(Array(0.8, 0.2), 0L)
    val simpleAls = new SimpleALS
    val k = 20
    val start = System.nanoTime()
    val (userFactors, prodFactors) = simpleAls.run(training, k = k, numBlocks = 1, numIterations = 0)
    val end = System.nanoTime()
    println("Time: " + (end - start) / 1e9)
    val predictionAndRatings = test.map(x => (x._1, (x._2, x._3))).join(userFactors).map { case (userId, ((prodId, rating), userFactor)) =>
      (prodId, (rating, userFactor))
    }.join(prodFactors).values.map { case ((rating, userFactor), prodFactor) =>
      (blas.sdot(k, userFactor, 1, prodFactor, 1), rating)
    }
    val mse = predictionAndRatings.map { case (pred, rating) =>
      val err = pred - rating
      err * err
    }.mean()
    val rmse = math.sqrt(mse)
    println(s"Test RMSE: $rmse.")
  }
}
