package controllers

import play.api.libs.json.{JsValue, Json}

case class ApiError(error:String) {

  def toJson: JsValue = Json.obj(("error", Json.toJsFieldJsValueWrapper(error)))

}
