package devoxx.dl4j.core.trainer

import java.io.File
import java.net.URL

import devoxx.dl4j.core.predictor.Constants
import org.apache.commons.io.{FileUtils, FilenameUtils}
import org.deeplearning4j.api.storage.impl.RemoteUIStatsStorageRouter
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator
import org.deeplearning4j.earlystopping.termination.{MaxEpochsTerminationCondition, ScoreImprovementEpochTerminationCondition}
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
import org.deeplearning4j.models.word2vec.Word2Vec
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{EmbeddingLayer, LSTM, RnnOutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.PerformanceListener
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions
import devoxx.dl4j.core.trainer.fetcher.Fetcher
import devoxx.dl4j.core.predictor.preprocessing.SentimentExampleIterator
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.stats.StatsListener
import org.deeplearning4j.ui.storage.InMemoryStatsStorage


object TrainMain {

  val OUTPUT_PATH: String = Constants.modelsPath + Constants.modelVersion + "test"

  @throws[Exception]
  def main(args: Array[String]): Unit = {
    Fetcher.downloadData(Constants.DATA_PATH, Constants.DATA_URL)


    val batchSize = 128
    val EMBEDDING_WIDTH = 300
    val nEpochs = 1
    val seed = 0
    Nd4j.getMemoryManager.setAutoGcWindow(10000)

    ////////////////////////////////////////////////////////////////////////////////
    /////////                                                              /////////
    /////////                       PREPROCESSING                          /////////
    /////////                                                              /////////
    ////////////////////////////////////////////////////////////////////////////////

    val wordVectors = WordVectorSerializer.loadStaticModel(new File(Constants.WORD_VECTORS_PATH))
    val train = new SentimentExampleIterator(Constants.DATA_PATH, wordVectors, batchSize, Constants.truncateReviewsToLength, true)
    val test = new SentimentExampleIterator(Constants.DATA_PATH, wordVectors, batchSize, Constants.truncateReviewsToLength, false)


    ////////////////////////////////////////////////////////////////////////////////
    /////////                                                              /////////
    /////////                      NEURAL NETWORK                          /////////
    /////////                                                              /////////
    ////////////////////////////////////////////////////////////////////////////////



    ////////////////////////////////////////////////////////////////////////////////
    /////////                                                              /////////
    /////////                         LISTENERS                            /////////
    /////////                                                              /////////
    ////////////////////////////////////////////////////////////////////////////////



    ////////////////////////////////////////////////////////////////////////////////
    /////////                                                              /////////
    /////////                      EARLY STOPPING                          /////////
    /////////                                                              /////////
    ////////////////////////////////////////////////////////////////////////////////


    ////////////////////////////////////////////////////////////////////////////////
    /////////                                                              /////////
    /////////                        PERSISTENCE                           /////////
    /////////                                                              /////////
    ////////////////////////////////////////////////////////////////////////////////


    ModelSerializer.writeModel(bestModel, OUTPUT_PATH, true)

  }

}

