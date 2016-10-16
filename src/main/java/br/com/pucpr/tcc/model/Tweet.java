package br.com.pucpr.tcc.model;

import java.util.Arrays;

/**
 * Created by douglas on 10/9/16.
 */
public class Tweet {

    private int label;

    private double[][] features;

    public int getLabel() {
        return label;
    }

    public void setLabel(int label) {
        this.label = label;
    }

    public double[][] getFeatures() {
        return features;
    }

    public void setFeatures(double[][] features) {
        this.features = features;
    }

    @Override
    public String toString() {
        return "Tweet{" +
                "label=" + label +
                ", features=" + Arrays.toString(features) +
                '}';
    }
}
