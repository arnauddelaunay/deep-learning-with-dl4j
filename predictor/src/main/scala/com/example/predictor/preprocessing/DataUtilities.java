package com.example.predictor.preprocessing;

import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream;

import java.io.*;

/**
 * Common data utility functions.
 *
 * @author fvaleri
 */
public class DataUtilities {


    /**
     * Extract a "tar.gz" file into a local folder.
     * @param inputPath Input file path.
     * @param outputPath Output directory path.
     * @throws IOException IO error.
     */
    public static void extractTarGz(String inputPath, String outputPath) throws IOException {
        if (inputPath == null || outputPath == null)
            return;
        final int bufferSize = 4096;
        if (!outputPath.endsWith("" + File.separatorChar))
            outputPath = outputPath + File.separatorChar;
        TarArchiveInputStream tais = new TarArchiveInputStream(
                new GzipCompressorInputStream(new BufferedInputStream(new FileInputStream(inputPath))));
        TarArchiveEntry entry;
        while ((entry = (TarArchiveEntry) tais.getNextEntry()) != null) {
            if (entry.isDirectory()) {
                new File(outputPath + entry.getName()).mkdirs();
            } else {
                int count;
                byte data[] = new byte[bufferSize];
                FileOutputStream fos = new FileOutputStream(outputPath + entry.getName());
                BufferedOutputStream dest = new BufferedOutputStream(fos, bufferSize);
                while ((count = tais.read(data, 0, bufferSize)) != -1) {
                    dest.write(data, 0, count);
                }
                dest.close();
            }
        }
    }

}
