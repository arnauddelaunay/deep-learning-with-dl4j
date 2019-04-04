package controllers

import play.api.libs.json.{JsValue, Json}
import play.api.mvc.{Action, Controller}
import java.awt.image.BufferedImage
import com.fasterxml.jackson.core.JsonParseException
import com.typesafe.config.ConfigFactory
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.cpu.nativecpu.NDArray
import play.Play
import play.api.libs.json._

object DRAWEndpoint extends Controller {

  def index = Action {

    Ok(views.html.drawmeacat.drawmeacat())

  }

  def recognise = Action { implicit request =>

    val body = request.body.asFormUrlEncoded.get("body")
    val in = parseData(body.head)
    in match {
      case Right(x) => BadRequest(ApiError(x).toJson)
      case Left(image) =>
        val recognise = MNISTResources.recognise(image)
        println(recognise._1)
        println(recognise._2.mkString(","))
        println()
        Ok (Json.prettyPrint(Json.obj("recognised" -> recognise._1, "results" -> recognise._2)))
    }
  }

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


}
