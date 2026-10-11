
import breeze.linalg._
import breeze.stats.distributions.Rand

object KMeansExample {

  def euclideanDistance(
                         point1: DenseVector[Double],
                         point2: DenseVector[Double]
                       ): Double = {

    val differences = point1 - point2
    math.sqrt(differences.data.map(v => v * v).sum)
  }

  def main(args: Array[String]): Unit = {

    val numSamples = 200
    val numFeatures = 2
    val k = 2
    val maxIterations = 100

    println("Generating a synthetic dataset...")

    val cluster1 = DenseMatrix.zeros[Double](numSamples / 2, numFeatures)
    val cluster2 = DenseMatrix.zeros[Double](numSamples / 2, numFeatures)

    for (i <- 0 until numSamples / 2) {
      cluster1(i, 0) = 1.0 + Rand.gaussian.draw()
      cluster1(i, 1) = 2.0 + Rand.gaussian.draw()

      cluster2(i, 0) = 5.0 + Rand.gaussian.draw()
      cluster2(i, 1) = 6.0 + Rand.gaussian.draw()
    }

    val data = DenseMatrix.vertcat(cluster1, cluster2)

    println(
      s"Generated dataset with ${data.rows} samples and ${data.cols} features."
    )

    var centroids = DenseMatrix.zeros[Double](k, numFeatures)

    val randomIndices = (0 until data.rows)
      .sortBy(_ => Rand.uniform.draw())
      .take(k)

    for (i <- 0 until k) {
      for (j <- 0 until numFeatures) {
        centroids(i, j) = data(randomIndices(i), j)
      }
    }

    println("\nInitial centroids:")
    println(centroids)

    var assignments = DenseVector.fill[Int](data.rows)(-1)

    var iteration = 0
    var converged = false

    while (iteration < maxIterations && !converged) {

      println(s"\n--- Iteration ${iteration + 1} ---")

      val newAssignments = DenseVector.zeros[Int](data.rows)

      // Assignment step
      for (i <- 0 until data.rows) {

        val point = data(i, ::).t
        var minDistance = Double.MaxValue
        var closestCentroidIndex = 0

        for (j <- 0 until k) {

          val centroid = centroids(j, ::).t
          val distance = euclideanDistance(point, centroid)

          if (distance < minDistance) {
            minDistance = distance
            closestCentroidIndex = j
          }
        }

        newAssignments(i) = closestCentroidIndex
      }

      // Check convergence
      converged = newAssignments == assignments
      assignments = newAssignments

      // Update centroids
      val newCentroids = DenseMatrix.zeros[Double](k, numFeatures)
      val clusterCounts = Array.fill[Int](k)(0)

      for (i <- 0 until data.rows) {

        val clusterId = assignments(i)
        clusterCounts(clusterId) += 1

        for (j <- 0 until numFeatures) {
          newCentroids(clusterId, j) += data(i, j)
        }
      }

      for (clusterId <- 0 until k) {

        if (clusterCounts(clusterId) > 0) {

          for (j <- 0 until numFeatures) {
            newCentroids(clusterId, j) =
              newCentroids(clusterId, j) / clusterCounts(clusterId).toDouble
          }

        } else {
          // Keep the previous centroid if a cluster is empty
          for (j <- 0 until numFeatures) {
            newCentroids(clusterId, j) = centroids(clusterId, j)
          }
        }
      }

      centroids = newCentroids

      println("Updated centroids:")
      println(centroids)

      iteration += 1
    }

    println("\n===== FINAL RESULTS =====")
    println(s"K-means iterations: $iteration")
    println(s"Converged: $converged")

    println("\nFinal centroids:")
    println(centroids)

    println("\nFinal cluster assignments:")
    println(assignments)

    for (clusterId <- 0 until k) {
      val count = assignments.data.count(_ == clusterId)
      println(s"Cluster $clusterId contains $count data points.")
    }
  }
}
