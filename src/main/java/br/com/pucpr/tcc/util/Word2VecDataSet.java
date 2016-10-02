package br.com.pucpr.tcc.util;

import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;

/**
 * Created by douglas on 10/2/16.
 */
public class Word2VecDataSet {

    private Logger log = LoggerFactory.getLogger(Word2VecDataSet.class);
    private final String trainFile;
    private final String outputFile;


    public Word2VecDataSet(String trainFile, String outputFile) {
        this.trainFile = trainFile;
        this.outputFile = outputFile;
    }

    public void train() throws IOException {
        log.info("Load & Vectorize Sentences....");
        // Strip white space before and after for each line
        SentenceIterator iter = new BasicLineIterator(trainFile);
        // Split on white spaces in the line to get words
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        log.info("Building model....");
        Word2Vec vec = new Word2Vec.Builder()
                .minWordFrequency(5)
                .iterations(1)
                .layerSize(100)
                .seed(42)
                .windowSize(5)
                .iterate(iter)
                .tokenizerFactory(t)
                .build();

        log.info("Fitting Word2Vec model....");
        vec.fit();

        log.info("Writing word vectors to text file....");

        // Write word vectors
        WordVectorSerializer.writeWordVectors(vec, outputFile);
    }

    public static WeightLookupTable lookupTable(String modelFile) {
        try {
            Word2Vec word2vec = WordVectorSerializer.readWord2Vec(new File(modelFile));
            return word2vec.lookupTable();
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }
}
