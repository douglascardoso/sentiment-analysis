package br.com.pucpr.tcc;

import br.com.pucpr.tcc.util.Model;

/**
 * Created by douglas on 10/10/16.
 */
public class Test {


    public static void main(String[] args) {

//        INDArray z = Nd4j.rand(10, 100);
//        INDArray o = Nd4j.rand(10, 1);
//
//        o.getRow(0).getColumn(0).assign(1);
//        o.getRow(1).getColumn(0).assign(2);
//
//        DataSet dataSet = new DataSet(z, o);
//
//        INDArray t = Nd4j.create();
//        System.out.println(Nd4j.concat(1, t, Nd4j.ones(1, 11)));

        Model model = new Model("/home/douglas/Desktop/word2vec.txt");
        model.generateDataSet("/home/douglas/PUCPR/tcc/sentiment-analysis/src/main/resources/semeval/train.tsv");

        //System.out.println(dataSet.getLabels());
        //System.out.println(dataSet.numExamples());

        //System.out.println(dataSet.get(10));

        //System.out.println();

//        dataSet.addFeatureVector(z);
//        dataSet.getFeatureMatrix().add(z);
//        System.out.println(dataSet.getFeatureMatrix());


    }

}
