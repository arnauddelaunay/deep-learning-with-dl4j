package devoxx.dl4j.core.trainer.drawmeacat

import devoxx.dl4j.core.predictor.Constants
import devoxx.dl4j.core.trainer.fetcher.DrawingFetcher

object TrainMain {

  def main(args: Array[String]): Unit = {
    DrawingFetcher.downloadData(Constants.DATA_PATH, Constants.SELECTED_CLASSES_FILE_PATH)

  }

}
