
import breeze.linalg.DenseVector

object KnnExample {

  case class DataPoint(
                        features: DenseVector[Double],
                        label: String
                      )

  def euclideanDistance(
                         point1: DenseVector[Double],
                         point2: DenseVector[Double]
                       ): Double = {

    require(
      point1.length == point2.length,
      "Vectors must have the same dimensions"
    )

    val differences = point1 - point2

    val squaredSum = differences.data
      .map(value => value * value)
      .sum

    math.sqrt(squaredSum)
  }

  def main(args: Array[String]): Unit = {

    val dataset = Seq(
      DataPoint(DenseVector(1.0, 2.0), "A"),
      DataPoint(DenseVector(1.5, 2.5), "A"),
      DataPoint(DenseVector(5.0, 6.0), "B"),
      DataPoint(DenseVector(5.5, 6.5), "B")
    )

    println("===== TRAINING DATA =====")

    dataset.foreach { point =>
      println(
        s"Features: ${point.features}, Label: ${point.label}"
      )
    }

    val newPoint = DenseVector(1.8, 2.3)

    println(s"\nNew data point: $newPoint")
    println("\n===== EUCLIDEAN DISTANCES =====")

    val distances = dataset.map { point =>
      val distance = euclideanDistance(
        newPoint,
        point.features
      )

      println(
        f"Label: ${point.label}%-3s Distance: $distance%.4f"
      )

      (point, distance)
    }

    val nearestNeighbor = distances.minBy(_._2)

    println("\n===== CLASSIFICATION RESULT =====")
    println(
      f"Nearest neighbor distance: ${nearestNeighbor._2}%.4f"
    )
    println(
      s"Predicted label: ${nearestNeighbor._1.label}"
    )
  }
}
