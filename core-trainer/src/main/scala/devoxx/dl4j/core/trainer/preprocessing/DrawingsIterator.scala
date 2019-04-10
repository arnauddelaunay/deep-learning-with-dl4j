package devoxx.dl4j.core.trainer.preprocessing

import java.io.File
import java.util.Random

import org.datavec.api.io.labels.ParentPathLabelGenerator
import org.datavec.api.split.FileSplit
import org.datavec.image.loader.NativeImageLoader
import org.datavec.image.recordreader.ImageRecordReader
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler

object DrawingsIterator {

  def apply(trainPath: String, testPath: String, height: Int, width: Int, channels: Int, numClasses: Int, batchSize: Int, seed: Int = 42):
  (RecordReaderDataSetIterator, RecordReaderDataSetIterator) = {
    val randomNumGen = new Random(seed)

    println("Data load...")
    val trainData = new File(trainPath)
    val trainSplit = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randomNumGen)
    val labelMaker = new ParentPathLabelGenerator()
    val trainRecordReader = new ImageRecordReader(height, width, channels, labelMaker)
    trainRecordReader.initialize(trainSplit)
    val train = new RecordReaderDataSetIterator(trainRecordReader, batchSize, 1, numClasses)

    val testData = new File(testPath)
    val testSplit = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, randomNumGen)
    val testRecordReader = new ImageRecordReader(height, width, channels, labelMaker)
    testRecordReader.initialize(testSplit)
    val test = new RecordReaderDataSetIterator(testRecordReader, batchSize, 1, numClasses)

    println("Data vectorization...")
    val imageScaler = new ImagePreProcessingScaler
    imageScaler.fit(train)
    train.setPreProcessor(imageScaler)
    test.setPreProcessor(imageScaler)
    (train, test)
  }

}
