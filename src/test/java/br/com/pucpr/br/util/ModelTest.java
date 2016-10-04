package br.com.pucpr.br.util;

import br.com.pucpr.tcc.util.Model;
import org.junit.Ignore;
import org.junit.Test;

/**
 * Created by douglas on 10/3/16.
 */
public class ModelTest {


    @Test
    @Ignore
    public void testModelGeneration() throws Exception {

        Model model = new Model("/home/douglas/Desktop/word2vec.txt", "/home/douglas/Desktop/trainTest.csv");

        String tweet = "positive\tMusical awareness: Great Big Beautiful Tomorrow has an ending, Now is the time does not";

        System.out.println((int) tweet.charAt(0));


        model.parseTweet(tweet, 100);
        //System.out.println();


    }

}
