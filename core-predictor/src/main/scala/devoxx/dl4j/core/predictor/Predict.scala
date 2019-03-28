package devoxx.dl4j.core.predictor

import java.io.File

import devoxx.dl4j.core.predictor.preprocessing.{SentimentExampleIterator, Utils}
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.indexing.NDArrayIndex

object Predict {

  val wordVectors = WordVectorSerializer.loadStaticModel(new File(
    Utils.getPathFromWebApp(Constants.WORD_VECTORS_PATH)))
  val test = new SentimentExampleIterator(
    Utils.getPathFromWebApp(Constants.DATA_PATH), wordVectors, 1,
    Constants.truncateReviewsToLength, false)

  val model: MultiLayerNetwork = ModelSerializer.restoreMultiLayerNetwork(
    Utils.getPathFromWebApp(Constants.modelsPath + Constants.modelVersion))

  def recognise(review: String): (String, Array[Double]) = {
    val features = test.loadFeaturesFromString(review, Constants.truncateReviewsToLength)
    val networkOutput = model.output(features)
    val timeSeriesLength = networkOutput.size(2)
    val probabilitiesAtLastWord = networkOutput.get(NDArrayIndex.point(0), NDArrayIndex.all, NDArrayIndex.point(timeSeriesLength - 1))
    val pos = probabilitiesAtLastWord.getDouble(0.asInstanceOf[Long])
    val neg = probabilitiesAtLastWord.getDouble(1.asInstanceOf[Long])
    val result = if (pos > neg) { "😃" } else { "😞" }
    (result, Array(pos, neg))
  }

}
