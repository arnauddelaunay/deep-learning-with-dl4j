package devoxx.dl4j.core.predictor

object Constants {

  val DATA_URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_simple.tar.gz"
  val DATA_PATH = "data/dl4j_w2vSentiment/"
  val WORD_VECTORS_PATH = "data/GoogleNews-vectors-negative300-SLIM.bin.gz"
  val truncateReviewsToLength = 256

  val modelsPath = "data/models/"
  val modelVersion = "lstm3.zip"
  val drawNetModelVersion = "drawNet_v0.zip"

  val IMAGE_SIZE = 28
  val SELECTED_CLASSES_FILE_PATH = "data/drawmeacat/classes.txt"

}
