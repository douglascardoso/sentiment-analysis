package br.com.pucpr.tcc.util;

import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.*;

/**
 * Created by douglas on 10/2/16.
 */
public class Model {

    private BufferedWriter output;
    private WeightLookupTable lookuptable;

    public Model(String word2vecModel, String outputPath) {
        lookuptable = Word2VecDataSet.lookupTable(word2vecModel);
        output = createFile(outputPath);
    }

    public void generateFile(String filePath) {
        BufferedReader file = openFile(filePath);

        try {
            for (int i = 0; file.ready(); i++) {
                String tweet = file.readLine();
                parseTweet(tweet);
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

    }

    private void parseTweet(String tweet) {
        StringBuilder stringBuilder = new StringBuilder();
        String[] tokens = tweet.split("\t");
        int label = parseClass(tokens[0]);
        stringBuilder.append(label);
        String[] words = tokens[1].split(" ");

        for (String word : words) {
            INDArray features = lookuptable.vector(word);
            if (features != null) {
                for (int i = 0; i < features.length(); i++) {
                    stringBuilder.append(",");
                    stringBuilder.append(features.getDouble(i));
                }
            }
        }

        try {
            output.write(stringBuilder.toString());
            output.newLine();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

    }

    private int parseClass(String clazz) {
        return ("positive".equals(clazz)) ? 1 : 0;
    }

    private BufferedReader openFile(String filePath) {
        try {
            BufferedReader bufferedReader = new BufferedReader(new FileReader(filePath));
            return bufferedReader;
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        }
    }

    private BufferedWriter createFile(String outputPath) {
        try {
            BufferedWriter output = new BufferedWriter(new FileWriter(outputPath));
            return output;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
