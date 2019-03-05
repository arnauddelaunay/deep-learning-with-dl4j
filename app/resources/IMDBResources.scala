package resources

import java.io.File

import com.typesafe.config.ConfigFactory
import com.vogonjeltz.machineInt.lib.dl4jModels.mnist.MnistModelApplication
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.indexing.NDArrayIndex
import play.api.libs.json._
import resources.sentimentdetection.SentimentExampleIterator
import resources.sentimentdetection.Word2VecSentimentRNN.{DATA_PATH, WORD_VECTORS_PATH, truncateReviewsToLength}

object IMDBResources {

  lazy val model: MultiLayerNetwork = ModelSerializer.restoreMultiLayerNetwork(ConfigFactory.load().getString("models.folder")+ConfigFactory.load().getString("models.lstm"))
  lazy val app = new MnistModelApplication(model)
  val wordVectors = WordVectorSerializer.loadStaticModel(new File(WORD_VECTORS_PATH))
  val test = new SentimentExampleIterator(DATA_PATH, wordVectors, 1, truncateReviewsToLength, false)


  def parseData(in: String): String = {
    val json = Json.parse(in)
    (json \ "review").as[String]
  }

  def recognise(review: String): (String, Array[Double]) = {
    val features = test.loadFeaturesFromString(review, truncateReviewsToLength)
    val networkOutput = model.output(features)
    val timeSeriesLength = networkOutput.size(2)
    val probabilitiesAtLastWord = networkOutput.get(NDArrayIndex.point(0), NDArrayIndex.all, NDArrayIndex.point(timeSeriesLength - 1))
    val pos = probabilitiesAtLastWord.getDouble(0.asInstanceOf[Long])
    val neg = probabilitiesAtLastWord.getDouble(1.asInstanceOf[Long])
    val result = if (pos > neg) { ":-)" } else { ":-(" }
    (result, Array(pos, neg))
  }

}
