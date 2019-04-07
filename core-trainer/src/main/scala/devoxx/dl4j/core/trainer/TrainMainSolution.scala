package devoxx.dl4j.core.trainer

import java.io.File
import java.util.Random

import org.datavec.api.io.labels.ParentPathLabelGenerator
import org.datavec.api.split.FileSplit
import org.datavec.image.loader.NativeImageLoader
import org.datavec.image.recordreader.ImageRecordReader
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator
import org.deeplearning4j.earlystopping.termination.{MaxEpochsTerminationCondition, ScoreImprovementEpochTerminationCondition}
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.{ConvolutionLayer, DenseLayer, OutputLayer, SubsamplingLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.PerformanceListener
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.stats.StatsListener
import org.deeplearning4j.ui.storage.InMemoryStatsStorage
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions

object TrainMainSolution {

  val BATCH_SIZE = 256
  val TRAIN_PATH = "exploration/data/labeled_images/train"
  val TEST_PATH = "exploration/data/labeled_images/test"
  val NUM_CLASSES = 10
  val OUTPUT_PATH = "data/models/drawingNet_v2.zip"

  val HEIGHT = 28
  val WIDTH = 28
  val CHANNELS = 1

  def main(args: Array[String]): Unit = {

    val seed = 1234
    val randomNumGen = new Random(seed)
    val nEpochs = 1

    println("Data load...")
    val trainData = new File(TRAIN_PATH)
    val trainSplit = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randomNumGen)
    val labelMaker = new ParentPathLabelGenerator()
    val trainRecordReader = new ImageRecordReader(HEIGHT, WIDTH, CHANNELS, labelMaker)
    trainRecordReader.initialize(trainSplit)
    val train = new RecordReaderDataSetIterator(trainRecordReader, BATCH_SIZE, 1, NUM_CLASSES)

    val testData = new File(TEST_PATH)
    val testSplit = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, randomNumGen)
    val testRecordReader = new ImageRecordReader(HEIGHT, WIDTH, CHANNELS, labelMaker)
    testRecordReader.initialize(testSplit)
    val test = new RecordReaderDataSetIterator(testRecordReader, BATCH_SIZE, 1, NUM_CLASSES)

    println("Data vectorization...")
    val imageScaler = new ImagePreProcessingScaler
    imageScaler.fit(train)
    train.setPreProcessor(imageScaler)
    test.setPreProcessor(imageScaler)

    println("Network configuration and training...")
    val networkConf = new NeuralNetConfiguration.Builder()
      .weightInit(WeightInit.XAVIER)
      .updater(new Adam(1e-3))
      .list
      .layer(new ConvolutionLayer.Builder(3, 3)
          .nIn(1)
          .nOut(16)
          .stride(1, 1)
          .activation(Activation.RELU)
          .build
      )
      .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
          .kernelSize(2, 2)
        .build
      )
      .layer(new ConvolutionLayer.Builder(3, 3)
        .nOut(32)
        .stride(1, 1)
        .activation(Activation.RELU)
        .build
      )
      .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
        .kernelSize(2, 2)
        .build
      )
      .layer(new ConvolutionLayer.Builder(3, 3)
        .nOut(64)
        .stride(1, 1)
        .activation(Activation.RELU)
        .build
      )
      .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
        .kernelSize(2, 2)
        .build
      )
      .layer(new DenseLayer.Builder()
          .nOut(128)
          .activation(Activation.RELU)
          .build
      )
      .layer(new OutputLayer.Builder()
          .nOut(NUM_CLASSES)
          .activation(Activation.SOFTMAX)
          .lossFunction(LossFunctions.LossFunction.MCXENT)
          .build
      )
      .setInputType(InputType.convolutionalFlat(HEIGHT, WIDTH, CHANNELS))
      .build

    val network = new MultiLayerNetwork(networkConf)
    network.init()
    println(s"Total num of params: ${network.numParams}")

    // evaluation while training (the score should go down)// evaluation while training (the score should go down)

    /*var i = 0
    while ( i < nEpochs ) {
      network.fit(train)
      println(s"Completed epoch $i")
      //val eval = network.evaluate(test)
      //println(eval.toString)
      train.reset()
      test.reset()
      i += 1
    }*/

    ////////////////////////////////////////////////////////////////////////////////
    /////////                                                              /////////
    /////////                         LISTENERS                            /////////
    /////////                                                              /////////
    ////////////////////////////////////////////////////////////////////////////////

    val uiServer = UIServer.getInstance
    val statsStorage = new InMemoryStatsStorage()
    uiServer.attach(statsStorage)
    network.setListeners(
      new StatsListener(statsStorage),
      new PerformanceListener(1)
    )

    ////////////////////////////////////////////////////////////////////////////////
    /////////                                                              /////////
    /////////                      EARLY STOPPING                          /////////
    /////////                                                              /////////
    ////////////////////////////////////////////////////////////////////////////////
    val earlyStopConf = new EarlyStoppingConfiguration.Builder[MultiLayerNetwork]()
      .epochTerminationConditions(
        new MaxEpochsTerminationCondition(15),
        new ScoreImprovementEpochTerminationCondition(5, 1e-3))
      .scoreCalculator(new DataSetLossCalculator(test, true))
      .evaluateEveryNEpochs(1)
      .build

    val trainer = new EarlyStoppingTrainer(earlyStopConf, network, train)

    ////////////////////////////////////////////////////////////////////////////////
    /////////                                                              /////////
    /////////                        PERSISTENCE                           /////////
    /////////                                                              /////////
    ////////////////////////////////////////////////////////////////////////////////

    val result = trainer.fit()
    val bestModel = result.getBestModel

    ModelSerializer.writeModel(bestModel, OUTPUT_PATH, true)
  }

}
