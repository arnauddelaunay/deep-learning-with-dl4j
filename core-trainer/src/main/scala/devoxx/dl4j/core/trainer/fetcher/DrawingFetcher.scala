package devoxx.dl4j.core.trainer.fetcher

import java.io.File
import java.net.URL
import org.apache.commons.io.FileUtils
import scala.io.Source

object DrawingFetcher {
  val GOOGLE_BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/simplified"
  val GOOGLE_DATA_EXT = ".ndjson"
  val QUICK_DRAW_DATA_PATH = "drawmeacat/data"

  def downloadData(dataPath: String, classesFilePath: String): Unit = {

    //Create directory if required
    val targetPath = s"$dataPath/$QUICK_DRAW_DATA_PATH"
    val directory = new File(targetPath)
    if (!directory.exists) directory.mkdir

    for (oneClass <- Source.fromFile(classesFilePath).getLines) {
      val classUrl = s"$GOOGLE_BASE_URL/$oneClass$GOOGLE_DATA_EXT"
      val downloadedFile = new File(s"$targetPath/$oneClass$GOOGLE_DATA_EXT")
      if (!downloadedFile.exists) {
        println(s"Downloading $oneClass data from $classUrl")
        FileUtils.copyURLToFile(new URL(classUrl), downloadedFile)
      }
      else println(s"Data for $oneClass already exists")
    }
  }
}
