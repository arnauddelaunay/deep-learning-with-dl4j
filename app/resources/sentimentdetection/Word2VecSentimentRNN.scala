package resources.sentimentdetection

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
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions

/** Example: Given a movie review (raw text), classify that movie review as either positive or negative based on the words it contains.
  * This is done by combining Word2Vec vectors and a recurrent neural network model. Each word in a review is vectorized
  * (using the Word2Vec model) and fed into a recurrent neural network.
  * Training data is the "Large Movie Review Dataset" from http://ai.stanford.edu/~amaas/data/sentiment/
  * This data set contains 25,000 training reviews + 25,000 testing reviews
  *
  * Process:
  * 1. Automatic on first run of example: Download data (movie reviews) + extract
  * 2. Load existing Word2Vec model (for example: Google News word vectors. You will have to download this MANUALLY)
  * 3. Load each each review. Convert words to vectors + reviews to sequences of vectors
  * 4. Train network
  *
  * With the current configuration, gives approx. 83% accuracy after 1 epoch. Better performance may be possible with
  * additional tuning.
  *
  * NOTE / INSTRUCTIONS:
  * You will have to download the Google News word vector model manually. ~1.5GB
  * The Google News vector model available here: https://code.google.com/p/word2vec/
  * Download the GoogleNews-vectors-negative300.bin.gz file
  * Then: set the WORD_VECTORS_PATH field to point to this location.
  *
  * @author Alex Black
  */
object Word2VecSentimentRNN {
  /** Data URL for downloading */
  val DATA_URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
  /** Location to save and extract the training/testing data */
  val DATA_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "dl4j_w2vSentiment/")
  /** Location (local file system) for the Google News vectors. Set this manually. */
  val WORD_VECTORS_PATH = "public/downloads/GoogleNews-vectors-negative300-SLIM.bin.gz"
  val OUTPUT_PATH = "public/models/lstm2.zip"
  val truncateReviewsToLength = 256

  @throws[Exception]
  def main(args: Array[String]): Unit = {
    if (WORD_VECTORS_PATH.startsWith("/PATH/TO/YOUR/VECTORS/")) throw new RuntimeException("Please set the WORD_VECTORS_PATH before running this example")
    //Download and extract data
    downloadData()
    val batchSize = 128
    //Number of examples in each minibatch
    val vectorSize = 300
    //Size of the word vectors. 300 in the Google News model
    val nEpochs = 1
    //Number of epochs (full passes of training data) to train on
    val seed = 0 //Seed for reproducibility
    Nd4j.getMemoryManager.setAutoGcWindow(10000) //https://deeplearning4j.org/workspaces

    //Set up network configuration
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
    //DataSetIterators for training and testing respectively
    val wordVectors = WordVectorSerializer.loadStaticModel(new File(WORD_VECTORS_PATH))
    val train = new SentimentExampleIterator(DATA_PATH, wordVectors, batchSize, truncateReviewsToLength, true)
    val test = new SentimentExampleIterator(DATA_PATH, wordVectors, batchSize, truncateReviewsToLength, false)
    println("Starting training")
    var i = 0
    while ( {
      i < nEpochs
    }) {
      net.fit(train)
      train.reset
      println("Epoch " + i + " complete. Starting evaluation:")
      //Run evaluation. This is on 25k reviews, so can take some time
      //val evaluation = net.evaluate(test)
      //println(evaluation)

      {
        i += 1;
        i - 1
      }
    }
    //After training: load a single example and generate predictions
    val firstPositiveReviewFile = new File(FilenameUtils.concat(DATA_PATH, "aclImdb/test/pos/0_10.txt"))
    val firstPositiveReview = FileUtils.readFileToString(firstPositiveReviewFile)
    val features = test.loadFeaturesFromString(firstPositiveReview, truncateReviewsToLength)
    val networkOutput = net.output(features)
    val timeSeriesLength = networkOutput.size(2)
    val probabilitiesAtLastWord = networkOutput.get(NDArrayIndex.point(0), NDArrayIndex.all, NDArrayIndex.point(timeSeriesLength - 1))
    val pPositive: Double = probabilitiesAtLastWord.getDouble(0.asInstanceOf[Long])
    val pNegative: Double = probabilitiesAtLastWord.getDouble(1.asInstanceOf[Long])
    println("\n\n-------------------------------")
    println("First positive review: \n" + firstPositiveReview)
    println("\n\nProbabilities at last time step:")
    println(s"p(positive): $pPositive")
    println(s"p(negative): $pNegative")
    println("----- Example complete -----")

    // Write model
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
