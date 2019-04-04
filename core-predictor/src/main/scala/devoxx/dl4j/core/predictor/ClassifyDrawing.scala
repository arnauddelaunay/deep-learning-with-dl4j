package devoxx.dl4j.core.predictor

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.shade.jackson.core.JsonParseException

object ClassifyDrawing {

  lazy val model: MultiLayerNetwork = Serialise.read(ConfigFactory.load().getString("models.folder")+ConfigFactory.load().getString("models.model"))
  lazy val app = new MnistModelApplication(model)

  def parseData(in: String): Either[MNISTImage, String] = {
    val json = try {Some(Json.parse(in))} catch {
      case e: JsonParseException => None
    }
    val image = json.map(_ \ "image").flatMap(_.asOpt[Array[Double]]).map(MNISTImage)
    image match {
      case None => Right("Image not found/could not be parsed")
      case Some(x) if x.image.length != 28*28 => Right("Image was not 28x28")
      case Some(x) => Left(x)
    }
  }

  def recognise(image: MNISTImage): (Int, Array[Double]) = {
    app.use(new NDArray(Utils.preProcess(image.toBufferedImage, 28, true).map(_.toFloat)))
  }


}
