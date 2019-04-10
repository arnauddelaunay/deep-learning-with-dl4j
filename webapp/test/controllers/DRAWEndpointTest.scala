package controllers

import org.scalatest.{FlatSpec, Matchers}

class DRAWEndpointTest extends FlatSpec with Matchers{


  "parseJson" should "extract array from json" in {
    // Given
    val expected = Array(1.0, 1.0, 0.0, 0.0)
    val bodyHead = "{\"image\": [1.0, 1.0, 0.0, 0.0]}"

    // When
    val actual = DRAWEndpoint.parseJson(bodyHead)

    //Then
    actual shouldEqual expected
  }
}
