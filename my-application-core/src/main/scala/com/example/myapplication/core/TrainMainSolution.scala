package com.example.myapplication.core

import java.io.File

import com.example.myapplication.core.fetcher.Fetcher
import com.example.myapplication.core.preprocessing.SentimentExampleIterator
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
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.stats.StatsListener
import org.deeplearning4j.ui.storage.InMemoryStatsStorage
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions


object TrainMainSolution {

  val OUTPUT_PATH = Constants.modelsPath + Constants.modelVersion

  @throws[Exception]
  def main(args: Array[String]): Unit = {
    Fetcher.downloadData(Constants.DATA_PATH, Constants.DATA_URL)


    val batchSize = 4
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

    val t = new DefaultTokenizerFactory

    t.setTokenPreProcessor(new CommonPreprocessor)

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
        .nIn(EMBEDDING_WIDTH)
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

    val uiServer = UIServer.getInstance
    val statsStorage = new InMemoryStatsStorage()
    uiServer.attach(statsStorage)
    net.setListeners(
      new StatsListener(statsStorage),
      new PerformanceListener(5)
    )

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
    /////////                        PERSISTENCE                           /////////
    /////////                                                              /////////
    ////////////////////////////////////////////////////////////////////////////////

    val result = trainer.fit()
    val bestModel = result.getBestModel

    ModelSerializer.writeModel(bestModel, OUTPUT_PATH, true)

  }

}

