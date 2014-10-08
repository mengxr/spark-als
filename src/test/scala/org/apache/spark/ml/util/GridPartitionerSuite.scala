package org.apache.spark.ml.util

import org.apache.spark.HashPartitioner
import org.scalatest.FunSuite

class GridPartitionerSuite extends FunSuite {
  test("GridPartitioner with two partitioners") {
    val numParts0 = 10
    val part0 = new HashPartitioner(numParts0)
    val numParts1 = 7
    val part1 = new HashPartitioner(numParts1)
    val gridPart = new GridPartitioner(Array(part0, part1))
    assert(gridPart.numPartitions === numParts0 * numParts1)
    for (key0 <- Seq(1, 0, 10, "a", "ab", ("c", 5));
         key1 <- Seq("some", -1, 0, ("d", "e"))) {
      val partition = gridPart.getPartition((key0, key1))
      assert(partition === part0.getPartition(key0) + part0.numPartitions * part1.getPartition(key1))
      assert(gridPart.indices(partition).toSeq === Seq(part0.getPartition(key0), part1.getPartition(key1)))
    }
  }
}
