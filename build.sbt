net.virtualvoid.sbt.graph.Plugin.graphSettings

organization := "com.github.mengxr"

name := "spark-als"

version := "0.1-SNAPSHOT"

scalaVersion := "2.10.4"

libraryDependencies += "org.apache.spark" %% "spark-mllib" % "1.1.0"

libraryDependencies += "org.scalatest" %% "scalatest" % "2.1.5" % Test
