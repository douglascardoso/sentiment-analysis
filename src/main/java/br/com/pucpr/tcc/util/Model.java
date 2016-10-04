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

    public void generateFile(String filePath, int window) {
        BufferedReader file = openFile(filePath);

        try {
            for (int i = 0; file.ready(); i++) {
                String tweet = file.readLine();
                parseTweet(tweet, window);
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

    }

    public void parseTweet(String tweet, int window) {
        StringBuilder stringBuilder = new StringBuilder();
        String[] tokens = tweet.split("\t");
        int label = parseClass(tokens[0]);
        stringBuilder.append(label);
        tokens[1] = cleanString(tokens[1]);
        String[] words = tokens[1].split(" ");

        for (int i = 0; i < window; i++) {
            if (i < words.length) {
                String word = words[i];
                INDArray features = lookuptable.vector(word);
                if (features != null) {
                    for (int j = 0; j < features.length(); j++) {
                        stringBuilder.append(",");
                        stringBuilder.append(features.getDouble(j));
                    }
                    System.out.println("aaa");
                } else {
                    for (int j = 0; j < 100; j++) {
                        stringBuilder.append(",");
                        stringBuilder.append(Math.random());
                    }
                }
            } else {
                for (int j = 0; j < 100; j++) {
                    stringBuilder.append(",");
                    stringBuilder.append(Math.random());
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

    private String cleanString(String str) {
        str = str.toLowerCase();
        StringBuilder stringBuilder = new StringBuilder();
        for (int i = 0; i < str.length(); i++) {
            int ascii = (int) str.charAt(i);
            if (ascii >= 97 && ascii <= 122) {
                stringBuilder.append(str.charAt(i));
            } else if (ascii == 32) {
                stringBuilder.append(str.charAt(i));
            } else if (ascii == 35) {
                stringBuilder.append(str.charAt(i));
            } else if (ascii == 64) {
                stringBuilder.append(str.charAt(i));
            }
        }
        return stringBuilder.toString();
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
