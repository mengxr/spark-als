package org.apache.spark.ml.util

import org.apache.spark.Partitioner

/**
 * Created by meng on 10/8/14.
 */
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
