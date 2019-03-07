package sentimentdetectionmodel

import java.io.File
import java.net.URL

import org.apache.commons.io.{FileUtils, FilenameUtils}
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
      .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
      .gradientNormalizationThreshold(1.0)
      .list()
      .layer(
    0,
    new LSTM.Builder()
      .nIn(300)
      .nOut(256)
      .activation(Activation.TANH)
      .build
    )
      .layer(1, new RnnOutputLayer.Builder()
      .activation(Activation.SOFTMAX)
      .lossFunction(LossFunctions.LossFunction.MCXENT)
      .nIn(256)
      .nOut(2)
      .build)
      .build
    val net = new MultiLayerNetwork(conf)
    net.init()
    net.setListeners(new ScoreIterationListener(1))

    ////////////////////////////////////////////////////////////////////////////////
    /////////                                                              /////////
    /////////                         TRAINING                             /////////
    /////////                                                              /////////
    ////////////////////////////////////////////////////////////////////////////////
    var i = 0
    while ( {
      i < nEpochs
    }) {
      net.fit(train)
      train.reset()
      println("Epoch " + i + " complete")
      i += 1
    }


    ////////////////////////////////////////////////////////////////////////////////
    /////////                                                              /////////
    /////////                        PERSISTENCE                           /////////
    /////////                                                              /////////
    ////////////////////////////////////////////////////////////////////////////////
    ModelSerializer.writeModel(net, OUTPUT_PATH, true)
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
