package org.apache.spark.ml

import com.github.fommil.netlib.BLAS.{getInstance => blas}
import org.apache.spark.SparkContext._
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
    val ratings = sc.textFile("/Users/meng/share/data/movielens/ml-1m/ratings.dat", 2)
        .map(s => Rating.parseRating(s, "::"))
    val Array(training, test) = ratings.randomSplit(Array(0.8, 0.2), 0L)
    val simpleAls = new SimpleALS
    val k = 10
    val start = System.nanoTime()
    val (userFactors, prodFactors) = simpleAls.run(training, k = k, numBlocks = 2, numIterations = 10, lambda = 0.1)
    val end = System.nanoTime()
    println("Time: " + (end - start) / 1e9)
    val predictionAndRatings = test.map(x => (x.user, (x.product, x.rating))).join(userFactors).map { case (userId, ((prodId, rating), userFactor)) =>
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
