package controllers.recognizer

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.factory.Nd4j

class Recognizer(modelPath: String) {

  var model: MultiLayerNetwork = ModelSerializer.restoreMultiLayerNetwork(modelPath)

  def recognise(image: Array[Double]): (Int, Array[Float]) = {
    val imgPreprocessed = Nd4j.create(image)
    val results = this.model.output(imgPreprocessed)
    (results.argMax().toIntVector.head, results.toFloatVector)
  }

  def reload(modelPath: String): Unit = {
    this.model = ModelSerializer.restoreMultiLayerNetwork(modelPath)
  }

}
