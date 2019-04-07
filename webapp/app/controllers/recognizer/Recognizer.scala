package controllers.recognizer

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.factory.Nd4j

object Recognizer {

  lazy val model: MultiLayerNetwork = ModelSerializer.restoreMultiLayerNetwork("../data/models/drawingNet_v2.zip")

  def recognise(image: Array[Double]): (Int, Array[Float]) = {
    val imgPreprocessed = Nd4j.create(image)
    val results = model.output(imgPreprocessed)
    (results.argMax().toIntVector.head, results.toFloatVector)
  }

}
