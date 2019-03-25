package com.example.myapplication.core.fetcher

import java.io.File
import java.net.URL

import org.apache.commons.io.FileUtils
import com.example.myapplication.core.preprocessing.DataUtilities

object Fetcher {
  def downloadData(targetPath: String, dataURL: String): Unit = {
    //Create directory if required
    val directory = new File(targetPath)
    if (!directory.exists) directory.mkdir

    //Download file:
    val archizePath = targetPath + "aclImdb_simple.tar.gz"
    val archiveFile = new File(archizePath)
    val extractedPath = targetPath + "aclImdb_simple"
    val extractedFile = new File(extractedPath)
    if (!archiveFile.exists) {
      println("Starting data download (80MB)...")
      FileUtils.copyURLToFile(new URL(dataURL), archiveFile)
      println("Data (.tar.gz file) downloaded to " + archiveFile.getAbsolutePath)
      //Extract tar.gz file to output directory
      DataUtilities.extractTarGz(archizePath, targetPath)
    }
    else {
      println("Data (.tar.gz file) already exists at " + archiveFile.getAbsolutePath)
      if (!extractedFile.exists) DataUtilities.extractTarGz(archizePath, targetPath)
      else println("Data (extracted) already exists at " + extractedFile.getAbsolutePath)
    }
  }
}
