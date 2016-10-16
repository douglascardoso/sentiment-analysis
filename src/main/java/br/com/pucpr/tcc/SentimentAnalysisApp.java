package br.com.pucpr.tcc;

import java.io.IOException;

import org.apache.commons.lang.StringUtils;

import br.com.pucpr.tcc.util.DataSetUtils;
import br.com.pucpr.tcc.util.Word2VecDataSet;

/**
 * Created by douglas on 9/29/16.
 */
public class SentimentAnalysisApp {

	public static final void main(String[] args) throws IOException {

		if (args.length > 0) {

			for (int i = 0; i < args.length; i++) {
				switch (args[i]) {
				case "-clear-corpus-twitter": {
					if (args.length < 5) {
						throw new RuntimeException("See usage");
					}
					long startTime = System.currentTimeMillis();

					String inputFile = ClassLoader.getSystemResource("corpus_twitter/training.csv").getPath();
					String outputFile = null;
					Integer workers = 10;
					for (int j = i + 1; j < (i + 4) && j < args.length; j += 2) { // ignoring
																					// the
																					// first
																					// argument
						if ("--i".equals(args[j])) {
							inputFile = args[j + 1];
						} else if ("--o".equals(args[j])) {
							outputFile = args[j + 1];
						} else if ("--w".equals(args[j])) {
							workers = Integer.parseInt(args[j + 1]);
						}
					}
					if (outputFile == null) {
						throw new RuntimeException("See usage");
					}
					DataSetUtils.cleanCorpusTwitter(inputFile, outputFile, workers);
					long endTime = System.currentTimeMillis();
					System.out.printf("Cleaning corpus twitter is finished with %d seconds\n",
							(endTime - startTime) / 1000);
				}
					break;
				case "-word2vec-model": {
					if (args.length < 5) {
						throw new RuntimeException("See usage");
					}

					String trainFile = null;
					String outputFile = null;

					for (int j = i + 1; j < (i + 4) && j < args.length; j += 2) { // ignoring
																					// the
																					// first
																					// argument
						if ("--i".equals(args[j])) {
							trainFile = args[j + 1];
						} else if ("--o".equals(args[j])) {
							outputFile = args[j + 1];
						}
					}
					if (trainFile == null || outputFile == null) {
						throw new RuntimeException("See usage");
					}

					Word2VecDataSet word2vec = new Word2VecDataSet(trainFile, outputFile);
					word2vec.train();
				}
					break;
				case "-train": {
					if (args.length < 6) {
						throw new RuntimeException("See usage");
					}

					String trainFile = null;
					String validationFile = null;
					String outputFile = null;
					String modelFile = null;

					for (int j = i + 1; j < (i + 6) && j < args.length; j += 2) {
						if ("--i".equals(args[j])) {
							trainFile = args[j + 1];
						} else if ("--o".equals(args[j])) {
							outputFile = args[j + 1];
						} else if ("--m".equals(args[j])) {
							modelFile = args[j + 1];
						} else if ("--v".equals(args[j])) {
							validationFile = args[j + 1];
						}
					}

					if (StringUtils.isEmpty(trainFile) || StringUtils.isEmpty(validationFile)
							|| StringUtils.isEmpty(outputFile) || StringUtils.isEmpty(modelFile)) {
						throw new RuntimeException("See usage");
					}

					GeneretaTraining main = new GeneretaTraining(modelFile, trainFile, validationFile);
					main.run(outputFile);
				}
					break;
				}
			}
		}

		return;
	}
}
