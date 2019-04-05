package devoxx.dl4j.core.predictor

import devoxx.dl4j.core.predictor.image.Image.Image
import devoxx.dl4j.core.predictor.preprocessing.Utils
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.cpu.nativecpu.NDArray
import play.api.libs.json.JsValue

object ClassifyDrawing {

  lazy val model: MultiLayerNetwork = ModelSerializer.restoreMultiLayerNetwork(
    Utils.getPathFromWebApp(Constants.modelsPath + Constants.drawNetModelVersion))
  //lazy val app = new MnistModelApplication(model)

  def parseData(json: Option[JsValue]): Either[Image, String] = {

    val image = json.map(_ \ "image").flatMap(_.asOpt[Array[Double]]).map(Image)
    image match {
      case None => Right("Image not found/could not be parsed")
      case Some(x) if x.image.length != 28*28 => Right("Image was not 28x28")
      case Some(x) => Left(x)
    }
  }

  def recognise(image: Image): (Int, Array[Double]) = {
    //app.use(new NDArray(Utils.preProcess(image.toBufferedImage, 28, true).map(_.toFloat)))
    (0, Array(1.0))
  }


}
