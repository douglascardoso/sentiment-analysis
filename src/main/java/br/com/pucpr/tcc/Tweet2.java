package br.com.pucpr.tcc;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Created by douglas on 10/13/16.
 */
public class Tweet2 {

    private int label;

    private INDArray features;

    public int getLabel() {
        return label;
    }

    public void setLabel(int label) {
        this.label = label;
    }

    public INDArray getFeatures() {
        return features;
    }

    public void setFeatures(INDArray features) {
        this.features = features;
    }
}
