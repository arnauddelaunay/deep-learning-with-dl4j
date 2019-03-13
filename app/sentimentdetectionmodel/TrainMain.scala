package sentimentdetectionmodel

import java.io.File
import java.net.URL

import org.apache.commons.io.{FileUtils, FilenameUtils}
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator
import org.deeplearning4j.earlystopping.termination.{MaxEpochsTerminationCondition, ScoreImprovementEpochTerminationCondition}
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
import org.deeplearning4j.nn.conf.layers.{LSTM, RnnOutputLayer}
import org.deeplearning4j.nn.conf.{GradientNormalization, NeuralNetConfiguration}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions
import sentimentdetectionmodel.preprocessing.{DataUtilities, SentimentExampleIterator}


object TrainMain {

  val DATA_URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
  val DATA_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "dl4j_w2vSentiment/")
  val WORD_VECTORS_PATH = "public/downloads/GoogleNews-vectors-negative300-SLIM.bin.gz"
  val OUTPUT_PATH = "public/models/lstm3.zip"
  val truncateReviewsToLength = 256

  @throws[Exception]
  def main(args: Array[String]): Unit = {
    val batchSize = 128
    val vectorSize = 300
    val nEpochs = 1
    val seed = 0
    Nd4j.getMemoryManager.setAutoGcWindow(10000)

    ////////////////////////////////////////////////////////////////////////////////
    /////////                                                              /////////
    /////////                       PREPROCESSING                          /////////
    /////////                                                              /////////
    ////////////////////////////////////////////////////////////////////////////////

    val wordVectors = WordVectorSerializer.loadStaticModel(new File(WORD_VECTORS_PATH))
    val train = new SentimentExampleIterator(DATA_PATH, wordVectors, batchSize, truncateReviewsToLength, true)
    val test = new SentimentExampleIterator(DATA_PATH, wordVectors, batchSize, truncateReviewsToLength, false)

    ////////////////////////////////////////////////////////////////////////////////
    /////////                                                              /////////
    /////////                      NEURAL NETWORK                          /////////
    /////////                                                              /////////
    ////////////////////////////////////////////////////////////////////////////////




    ////////////////////////////////////////////////////////////////////////////////
    /////////                                                              /////////
    /////////                      EARLY STOPPING                          /////////
    /////////                                                              /////////
    ////////////////////////////////////////////////////////////////////////////////




    ////////////////////////////////////////////////////////////////////////////////
    /////////                                                              /////////
    /////////                         TRAINING                             /////////
    /////////                                                              /////////
    ////////////////////////////////////////////////////////////////////////////////




    ////////////////////////////////////////////////////////////////////////////////
    /////////                                                              /////////
    /////////                        PERSISTENCE                           /////////
    /////////                                                              /////////
    ////////////////////////////////////////////////////////////////////////////////



  }

}
