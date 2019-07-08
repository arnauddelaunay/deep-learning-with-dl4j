package devoxx.dl4j.core.trainer

import devoxx.dl4j.core.trainer.preprocessing.DrawingsIterator
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
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions

object TrainMain {

  val BATCH_SIZE = 256
  val TRAIN_PATH = "data/images/train"
  val TEST_PATH = "data/images/test"
  val NUM_CLASSES = 10
  val OUTPUT_PATH = "data/models/drawingNet.zip"

  val HEIGHT = 28
  val WIDTH = 28
  val CHANNELS = 1

  def main(args: Array[String]): Unit = {

    val nEpochs = 5

    val (train, test) = DrawingsIterator(TRAIN_PATH, TEST_PATH, HEIGHT, WIDTH, CHANNELS, NUM_CLASSES, BATCH_SIZE)

    ////////////////////////////////////////////////////////////////////////////////
    /////////                                                              /////////
    /////////                      EARLY STOPPING                          /////////
    /////////                                                              /////////
    ////////////////////////////////////////////////////////////////////////////////




    ////////////////////////////////////////////////////////////////////////////////
    /////////                                                              /////////
    /////////                         LISTENERS                            /////////
    /////////                                                              /////////
    ////////////////////////////////////////////////////////////////////////////////




    ////////////////////////////////////////////////////////////////////////////////
    /////////                                                              /////////
    /////////                        PERSISTENCE                           /////////
    /////////                                                              /////////
    ////////////////////////////////////////////////////////////////////////////////


    //ModelSerializer.writeModel(bestModel, OUTPUT_PATH, true)
  }

}
