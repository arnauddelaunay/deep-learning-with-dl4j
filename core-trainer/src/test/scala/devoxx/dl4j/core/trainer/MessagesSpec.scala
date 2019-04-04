package devoxx.dl4j.core.trainer

import org.specs2.mutable.SpecificationWithJUnit

class MessagesSpec extends SpecificationWithJUnit {

  "Messages" should {
    "return hello world" in {
      Messages.helloWorld must_== "Hello world!"
    }
  }
}