package controllers

import com.fasterxml.jackson.core.JsonParseException
import devoxx.dl4j.core.predictor.ClassifyDrawing
import play.api.libs.json.{JsValue, Json}
import play.api.mvc.{Action, Controller}


object DRAWEndpoint extends Controller {

  def index = Action {

    Ok(views.html.drawmeacat.drawmeacat())

  }

  def recognise = Action { implicit request =>

    val body = request.body.asFormUrlEncoded.get("body")
    val json: Option[JsValue] = try {Some(Json.parse(body.head))} catch {
      case e: JsonParseException => None
    }
    val in = ClassifyDrawing.parseData(json)
    in match {
      case Right(x) => BadRequest(ApiError(x).toJson)
      case Left(image) =>
        val recognise = ClassifyDrawing.recognise(image)
        println(recognise._1)
        println(recognise._2.mkString(","))
        println()
        Ok (Json.prettyPrint(Json.obj("recognised" -> recognise._1, "results" -> recognise._2)))
    }
  }


}
