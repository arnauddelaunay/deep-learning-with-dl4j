package devoxx.dl4j.core.predictor.image

import java.awt.image.BufferedImage
import devoxx.dl4j.core.predictor.Constants

object Image {
  val PIXEL_INT = 255

  case class Image(image: Array[Double]) {

    def toBufferedImage: BufferedImage = {
      val bufImage = new BufferedImage(Constants.IMAGE_SIZE, Constants.IMAGE_SIZE, BufferedImage.TYPE_INT_RGB)
      val intImage = image.map(X => (PIXEL_INT - X * PIXEL_INT).toInt)
      for (x <- Range(0, Constants.IMAGE_SIZE)) {
        for (y <- Range(0, Constants.IMAGE_SIZE)) {
          val value = intImage(y * Constants.IMAGE_SIZE + x) << 16 | intImage(y * Constants.IMAGE_SIZE + x) << 8 | intImage(y * Constants.IMAGE_SIZE + x)
          bufImage.setRGB(x, y, value)
        }
      }
      bufImage
    }
  }

}
