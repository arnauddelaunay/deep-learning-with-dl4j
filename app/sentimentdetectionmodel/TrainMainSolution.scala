package sentimentdetectionmodel

import java.io.File
import java.net.URL

import org.apache.commons.io.{FileUtils, FilenameUtils}
import org.deeplearning4j.api.storage.impl.RemoteUIStatsStorageRouter
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator
import org.deeplearning4j.earlystopping.termination.{MaxEpochsTerminationCondition, ScoreImprovementEpochTerminationCondition}
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{LSTM, RnnOutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.PerformanceListener
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.stats.StatsListener
import org.deeplearning4j.ui.storage.InMemoryStatsStorage
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions
import sentimentdetectionmodel.preprocessing.{DataUtilities, SentimentExampleIterator}


object TrainMainSolution {

  val DATA_URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
  val DATA_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "dl4j_w2vSentiment/")
  val WORD_VECTORS_PATH = "public/downloads/GoogleNews-vectors-negative300-SLIM.bin.gz"
  val OUTPUT_PATH = "public/models/lstm3.zip"
  val truncateReviewsToLength = 256

  @throws[Exception]
  def main(args: Array[String]): Unit = {
    if (WORD_VECTORS_PATH.startsWith("/PATH/TO/YOUR/VECTORS/")) throw new RuntimeException("Please set the WORD_VECTORS_PATH before running this example")
    downloadData()

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

    val conf = new NeuralNetConfiguration.Builder()
      .seed(seed)
      .updater(new Adam(5e-3))
      .l2(1e-5)
      .weightInit(WeightInit.XAVIER)
      .list
      .layer(0, new LSTM.Builder()
                    .nIn(vectorSize)
                    .nOut(256)
                    .activation(Activation.TANH)
                    .build
      )
      .layer(1, new RnnOutputLayer.Builder()
                    .activation(Activation.SOFTMAX)
                    .lossFunction(LossFunctions.LossFunction.MCXENT)
                    .nIn(256)
                    .nOut(2)
                    .build
      ).build

    val net = new MultiLayerNetwork(conf)
    net.init()


    ////////////////////////////////////////////////////////////////////////////////
    /////////                                                              /////////
    /////////                         LISTENERS                            /////////
    /////////                                                              /////////
    ////////////////////////////////////////////////////////////////////////////////

    /*val remoteUIRouter = new RemoteUIStatsStorageRouter("http://127.0.0.1:9001")
    net.setListeners(
      new StatsListener(remoteUIRouter),
      new PerformanceListener(5)
    )*/

    ////////////////////////////////////////////////////////////////////////////////
    /////////                                                              /////////
    /////////                      EARLY STOPPING                          /////////
    /////////                                                              /////////
    ////////////////////////////////////////////////////////////////////////////////


    val earlyStopConf = new EarlyStoppingConfiguration.Builder[MultiLayerNetwork]()
      .epochTerminationConditions(
        new MaxEpochsTerminationCondition(100),
        new ScoreImprovementEpochTerminationCondition(5, 5e-7))
      .scoreCalculator(new DataSetLossCalculator(test, true))
      .evaluateEveryNEpochs(1)
      .build
    val trainer = new EarlyStoppingTrainer(earlyStopConf, net, train)


    ////////////////////////////////////////////////////////////////////////////////
    /////////                                                              /////////
    /////////                         TRAINING                             /////////
    /////////                                                              /////////
    ////////////////////////////////////////////////////////////////////////////////
    val bestModel = trainer.fit().getBestModel


    ////////////////////////////////////////////////////////////////////////////////
    /////////                                                              /////////
    /////////                        PERSISTENCE                           /////////
    /////////                                                              /////////
    ////////////////////////////////////////////////////////////////////////////////
    ModelSerializer.writeModel(bestModel, OUTPUT_PATH, true)
  }

  @throws[Exception]
  def downloadData(): Unit = { //Create directory if required
    val directory = new File(DATA_PATH)
    if (!directory.exists) directory.mkdir
    //Download file:
    val archizePath = DATA_PATH + "aclImdb_v1.tar.gz"
    val archiveFile = new File(archizePath)
    val extractedPath = DATA_PATH + "aclImdb"
    val extractedFile = new File(extractedPath)
    if (!archiveFile.exists) {
      println("Starting data download (80MB)...")
      FileUtils.copyURLToFile(new URL(DATA_URL), archiveFile)
      println("Data (.tar.gz file) downloaded to " + archiveFile.getAbsolutePath)
      //Extract tar.gz file to output directory
      DataUtilities.extractTarGz(archizePath, DATA_PATH)
    }
    else { //Assume if archive (.tar.gz) exists, then data has already been extracted
      println("Data (.tar.gz file) already exists at " + archiveFile.getAbsolutePath)
      if (!extractedFile.exists) DataUtilities.extractTarGz(archizePath, DATA_PATH)
      else println("Data (extracted) already exists at " + extractedFile.getAbsolutePath)
    }
  }
}
