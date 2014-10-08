package org.apache.spark.ml.util

import org.apache.spark.Partitioner

class GridPartitioner(val partitioners: Array[Partitioner]) extends Partitioner {

  require(partitioners.size > 1)

  private val cumProds: Array[Int] = partitioners.map(_.numPartitions).scanLeft(1)(_ * _)

  override val numPartitions: Int = cumProds.last

  override def getPartition(key: Any): Int = {
    val keys = key.asInstanceOf[Product]
    require(keys.productArity == partitioners.size)
    var partition = 0
    var i = 0
    while (i < partitioners.size) {
      partition += partitioners(i).getPartition(keys.productElement(i)) * cumProds(i)
      i += 1
    }
    partition
  }

  override def equals(other: Any): Boolean = {
    other match {
      case o: GridPartitioner =>
        partitioners.zip(o.partitioners).forall(p => p._1 == p._2)
      case _ =>
        false
    }
  }

  def indices(partition: Int): Array[Int] = {
    val output = new Array[Int](partitioners.size)
    var cur = partition
    var i = 0
    while (i < partitioners.size) {
      output(i) = cur % partitioners(i).numPartitions
      cur /= partitioners(i).numPartitions
      i += 1
    }
    output
  }
}
