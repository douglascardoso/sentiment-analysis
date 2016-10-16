package br.com.pucpr.tcc.util;

import br.com.pucpr.tcc.model.Tweet;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Created by douglas on 10/2/16.
 */
public class Model {

    private BufferedWriter output;
    private WeightLookupTable lookuptable;

    private List<Tweet> tweets = new ArrayList<>();

    public Model(String word2vecModel) {
        lookuptable = Word2VecDataSet.lookupTable(word2vecModel);
    }

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


            final int[] count = {0};
            tweets.forEach(tweet -> {
                count[0] += tweet.getFeatures()[1].length;
            });
            System.out.println(Double.valueOf(count[0] / tweets.size()));
            writeToFile(20);

            output.close();

        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public DataSet generateDataSet(String filePath) {
        BufferedReader file = openFile(filePath);

        try {
            List<INDArray> list = new ArrayList<>();
            List<Integer> labels = new ArrayList<>();

            for (int i = 0; file.ready(); i++) {
                String tweet = file.readLine();
                Map.Entry<Integer, INDArray> entry = getTweet(tweet);
                list.add(entry.getValue());
                labels.add(entry.getKey());
            }

            INDArray dataSetArray = Nd4j.create(list.size(), 1024);
            INDArray labelsArray = Nd4j.create(labels.size(), 1);

            for (int i = 0; i < list.size(); i++) {
                INDArray row = dataSetArray.getRow(i);
                INDArray features = list.get(i);
                if (features.getRow(0).length() < 1024) continue;
                for (int j = 0; j < 1024; j++) {
                    dataSetArray.getRow(i).getColumn(j).assign(features.getRow(0).getColumn(j));
                }
            }

            for (int i = 0; i < labels.size(); i++) {
                labelsArray.getRow(i).getColumn(0).assign(labels.get(i));
            }

            DataSet dataSet = new DataSet(dataSetArray, labelsArray);

            return dataSet;

        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    //arrumar essa logica
    private void writeToFile(int sentenceLength) {
        for (Tweet tweet : tweets) {
            Double[] vector = toOneVector(tweet, 1024);

            if (vector == null) continue;
            /*double[][] features = tweet.getFeatures();

            if (sentenceLength > features[0].length) continue;*/

            StringBuilder str = new StringBuilder();
            str.append(tweet.getLabel());

//            for (int i = 0; i < features.length; i++) {
//                for (int j = 0; j < sentenceLength; j++) {
//                    str.append(",");
//                    str.append(features[i][j]);
//                }
//            }

            for (int i = 0; i < vector.length; i++) {
                str.append(",");
                str.append(vector[i]);
            }

            try {
                output.write(str.toString());
                output.newLine();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    private Double[] toOneVector(Tweet tweet, int size) {
        double[][] features = tweet.getFeatures();

        System.out.println(features[0].length * features[1].length);

        if ((features[0].length * features[1].length) < size)
            return null;

        Double[] oneVector = new Double[size];
        int index = 0;
        for (int i = 0; i < features.length; i++) {
            if (index > size - 1) break;
            for (int j = 0; j < features[i].length; j++) {
                if (index > size - 1) break;
                oneVector[index] = features[i][j];
                index += 1;
            }
        }

        return oneVector;
    }

    public void parseTweet(String line, int window) {
        Tweet tweet = new Tweet();

        String[] tokens = line.split("\t");

        // Setting the label
        int label = parseClass(tokens[0]);
        tweet.setLabel(label);

        // Parsing the words
        String[] words = cleanString(tokens[1]).split(" ");

        double[][] features = new double[window][words.length];

        for (int i = 0; i < words.length; i++) {
            String word = words[i];
            INDArray array = lookuptable.vector(word);

            double[] res = toDoubleArray(array, window);

            for (int j = 0; j < res.length; j++) {
                features[j][i] = res[j];
            }
        }

//        for (int j = 0; j < words.length; j++) {
//            String word = words[i];
//            INDArray array = lookuptable.vector(word);
//            features[i] = toDoubleArray(array, window);
//        }

        tweet.setFeatures(features);
        tweets.add(tweet);
    }

    public Map.Entry<Integer, INDArray> getTweet(String line) {

        String[] tokens = line.split("\t");
        Integer label = Integer.valueOf(parseClass(tokens[0]));

        String[] words = cleanString(tokens[1]).split(" ");

        INDArray features = null;

        for (int i = 0; i < words.length; i++) {
            String word = words[i];
            INDArray array = lookuptable.vector(word);
            if (array == null) continue;
            if (features == null) features = array;
            else features = Nd4j.concat(1, features, array);
        }

        Map.Entry<Integer, INDArray> tweet = new AbstractMap.SimpleEntry<Integer, INDArray>(label, features);

        return tweet;
    }

    private double[] toDoubleArray(INDArray array, int window) {

        if (array != null) {
            double[] features = new double[array.length()];
            for (int i = 0; i < array.length(); i++) {
                features[i] = array.getDouble(i);
            }
            return features;
        }
        return randomFeatures(window);
    }

    private double[] randomFeatures(int window) {
        double[] features = new double[window];
        for (int i = 0; i < window; i++) {
            features[i] = Math.random();
        }
        return features;
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

    public List<Tweet> getTweets() {
        return tweets;
    }
}
