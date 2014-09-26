package org.apache.spark.ml.util

/**
 * Abstraction for sorting an arbitrary input buffer of data. This interface requires determining
 * the sort key for a given element index, as well as swapping elements and moving data from one
 * buffer to another.
 *
 * Example format: an array of numbers, where each element is also the key.
 * See [[KVArraySortDataFormat]] for a more exciting format.
 *
 * This trait extends Any to ensure it is universal (and thus compiled to a Java interface).
 *
 * @tparam K Type of the sort key of each element
 * @tparam Buffer Internal data structure used by a particular format (e.g., Array[Int]).
 */
// TODO: Making Buffer a real trait would be a better abstraction, but adds some complexity.
trait SortDataFormat[Buffer] extends Any {

  /** Return the sort key for the element at the given index. */
  protected def getKey(data: Buffer, pos: Int): Int

  /** Swap two elements. */
  protected def swap(data: Buffer, pos0: Int, pos1: Int): Unit

  /** Copy a single element from src(srcPos) to dst(dstPos). */
  protected def copyElement(src: Buffer, srcPos: Int, dst: Buffer, dstPos: Int): Unit

  /**
   * Copy a range of elements starting at src(srcPos) to dst, starting at dstPos.
   * Overlapping ranges are allowed.
   */
  protected def copyRange(src: Buffer, srcPos: Int, dst: Buffer, dstPos: Int, length: Int): Unit

  /**
   * Allocates a Buffer that can hold up to 'length' elements.
   * All elements of the buffer should be considered invalid until data is explicitly copied in.
   */
  protected def allocate(length: Int): Buffer
}

trait IntComparator {
  def compare(o1: Int, o2: Int): Int
}
