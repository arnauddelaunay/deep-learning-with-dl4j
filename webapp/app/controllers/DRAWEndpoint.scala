package controllers

import controllers.recognizer.Recognizer
import play.api.libs.json.Json
import play.api.mvc.{Action, Controller}

object DRAWEndpoint extends Controller {

  val zeros: String = Array.fill(28*28)(0.0).mkString(",")
  val defaultBody = s"""{"image": [$zeros]}"""

  val classes: Seq[String] = Seq("airplane", "bicycle", "cat", "chair", "cup", "ladder", "snake", "star", "sun", "table")
  val emojis: Map[String, String] = Map("airplane" -> "✈️", "bicycle" -> "\uD83D\uDEB2", "cat" -> "\uD83D\uDC08",
    "chair" -> "\uD83D\uDCBA", "cup" -> "☕", "ladder" -> "\uD83E\uDDD7", "snake" -> "\uD83D\uDC0D",
    "star" -> "\uD83C\uDF1F", "sun" -> "☀️", "table" -> "┳━┳")

  val defaultModelPath = "../data/models/drawingNet.zip"
  val bestModelPath = "../data/models/drawingNet_v2.zip"
  val recognizer = new Recognizer(defaultModelPath)

  def index = Action {

    Ok(views.html.drawmeacat.drawmeacat())

  }

  def recognise = Action { implicit request =>

    val body = request.body.asFormUrlEncoded.get("body")
    val imgArray = parseJson(body.head).map(x => 1.0 - x)
    val recognise = recognizer.recognise(imgArray)
    println(recognise._1)
    println(recognise._2.mkString(","))
    Ok (Json.prettyPrint(Json.obj(
      "classIndex" -> recognise._1,
      "recognised" -> classes(recognise._1),
      "results" -> recognise._2,
      "classes" -> classes,
      "emojis" -> emojis
    )))
  }

  def parseJson(json: String): Array[Double] = {
    (Json.parse(json) \ "image").as[Array[Double]]
  }

  def reload = Action {
    this.recognizer.reload(defaultModelPath)
    Ok("model reloaded")
  }

  def reloadBest = Action {
    this.recognizer.reload(bestModelPath)
    Ok("best model reloaded")
  }


}
