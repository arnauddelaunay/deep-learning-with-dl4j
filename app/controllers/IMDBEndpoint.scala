package controllers

import play.api.libs.json.Json
import play.api.mvc.{Action, Controller}
import resources.{ApiError, IMDBResources}

/**
  * Created by Freddie on 10/06/2017.
  */
class IMDBEndpoint extends Controller {

  def index = Action {

    Ok(views.html.imdb.imdb_home())

  }

  def recognise = Action { implicit request =>

    val body = request.body.asFormUrlEncoded.get("body")
    val review = IMDBResources.parseData(body.head)
    val recognise = IMDBResources.recognise(review)
      println(recognise._1)
      println(recognise._2.mkString(","))
      println()
      Ok (Json.prettyPrint(Json.obj("recognised" -> recognise._1, "results" -> recognise._2)))

  }

}
