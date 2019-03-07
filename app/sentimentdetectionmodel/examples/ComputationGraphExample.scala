package sentimentdetectionmodel.examples

import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.graph.MergeVertex
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.graph.ComputationGraph
import org.nd4j.linalg.learning.config.Sgd

object ComputationGraphExample {

  val CGConf = new NeuralNetConfiguration.Builder()
    .updater(new Sgd(0.01))
    .graphBuilder()
    .addInputs("input1", "input2")
    .addLayer(
      "L1",
      new DenseLayer.Builder()
        .nIn(3)
        .nOut(4)
        .build(),
      "input1"
    )
    .addLayer(
      "L2",
      new DenseLayer.Builder()
        .nIn(3)
        .nOut(4)
        .build(),
      "input2"
    )
    .addVertex(
      "merge",
      new MergeVertex(),
      "L1", "L2"
    )
    .addLayer(
      "out",
      new OutputLayer.Builder()
        .nIn(4+4)
        .nOut(3)
        .build(),
      "merge")
    .setOutputs("out")
    .build()
  val computationGraph = new ComputationGraph(CGConf)
  computationGraph.init()

}
