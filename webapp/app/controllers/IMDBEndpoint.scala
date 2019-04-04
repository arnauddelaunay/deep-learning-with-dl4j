package controllers

import play.api.libs.json.Json
import play.api.mvc.{Action, Controller}

import devoxx.dl4j.core.predictor.Predict

object IMDBEndpoint extends Controller {

  def index = Action {

    Ok(views.html.imdb.imdb_home())

  }

  def recognise = Action { implicit request =>

    val body = request.body.asFormUrlEncoded.get("body")
    val review = parseData(body.head)
    val recognise = Predict.recognise(review)
    println(recognise._1)
    println(recognise._2.mkString(","))
    println()
    Ok (Json.prettyPrint(Json.obj("recognised" -> recognise._1, "results" -> recognise._2)))

  }

  def parseData(body: String): String = {
    val json = Json.parse(body)
    (json \ "review").as[String]
  }

}
