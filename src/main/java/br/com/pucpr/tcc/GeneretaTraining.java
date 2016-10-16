package br.com.pucpr.tcc;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.SamplingDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import br.com.pucpr.tcc.model.Tweet;
import br.com.pucpr.tcc.util.Word2VecDataSet;

/**
 * Created by douglas on 10/13/16.
 */
public class GeneretaTraining {

	private static final Logger log = LoggerFactory.getLogger(GeneretaTraining.class);

	private WeightLookupTable<?> lookuptable;

	private String trainPath;

	private String validationPath;

	public GeneretaTraining(String word2vecPath) {
		lookuptable = Word2VecDataSet.lookupTable(word2vecPath);
	}

	public GeneretaTraining(String word2vecModelPath, String trainPath, String validationPath) {
		lookuptable = Word2VecDataSet.lookupTable(word2vecModelPath);
		this.trainPath = trainPath;
		this.validationPath = validationPath;
	}

	public static void main(String[] args) throws IOException {

		GeneretaTraining main = new GeneretaTraining("/home/douglas/datasets/word2vec.txt",
				"/home/douglas/sentiment-analysis/src/main/resources/semeval/train.tsv",
				"/home/douglas/sentiment-analysis/src/main/resources/semeval/validation.tsv");
		main.run("");
	}

	public void test(String modelPath, String testPath) throws IOException {
		int outputNum = 2;
		int batchSize = 150;

		List<Tweet> testLines = readFile(testPath);

		INDArray testArray = Nd4j.zeros(testLines.size(), 10000);
		INDArray testOutcomes = Nd4j.zeros(testLines.size(), 2);

		DataSet test = new DataSet(testArray, testOutcomes);
		DataSetIterator testIterator = new SamplingDataSetIterator(test, batchSize, testLines.size());

		// Test
		for (int i = 0; i < testLines.size(); i++) {
			Tweet tweet = testLines.get(i);
			testOutcomes.getRow(i).getColumn(tweet.getLabel()).assign(1);
			testArray.getRow(i).assign(tweet.getFeatures());
		}

		log.info("Loading model");
		MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(modelPath);

		log.info("Evaluate model....");
		Evaluation eval = new Evaluation(outputNum);
		while (testIterator.hasNext()) {
			DataSet ds = testIterator.next();

			INDArray output = model.output(ds.getFeatureMatrix(), false);
			eval.eval(ds.getLabels(), output);
		}
		log.info(eval.stats());
	}

	public void run(String outputFile) throws IOException {
		Scanner scanner = new Scanner(System.in);
		List<Tweet> trainLines = readFile(trainPath);
		List<Tweet> testLines = readFile(validationPath);

		INDArray trainArray = Nd4j.zeros(trainLines.size(), 10000);
		INDArray trainOutcomes = Nd4j.zeros(trainLines.size(), 2);

		INDArray testArray = Nd4j.zeros(testLines.size(), 10000);
		INDArray testOutcomes = Nd4j.zeros(testLines.size(), 2);

		// Train
		for (int i = 0; i < trainLines.size(); i++) {
			Tweet tweet = trainLines.get(i);
			trainOutcomes.getRow(i).getColumn(tweet.getLabel()).assign(1);
			trainArray.getRow(i).assign(tweet.getFeatures());
		}

		// Validation
		for (int i = 0; i < testLines.size(); i++) {
			Tweet tweet = testLines.get(i);
			testOutcomes.getRow(i).getColumn(tweet.getLabel()).assign(1);
			testArray.getRow(i).assign(tweet.getFeatures());
		}

		System.gc();

		int nChannels = 1;
		int outputNum = 2;
		int batchSize = 150;
		int nEpochs = 10;
		int iterations = 1;
		int seed = 123;

		DataSet train = new DataSet(trainArray, trainOutcomes);
		DataSet test = new DataSet(testArray, testOutcomes);

		train.shuffle();
		test.shuffle();

		DataSetIterator trainIterator = new SamplingDataSetIterator(train, batchSize, trainLines.size());
		DataSetIterator testIterator = new SamplingDataSetIterator(test, batchSize, testLines.size());

		MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder().seed(seed).iterations(iterations)
				.regularization(true).l2(0.0005).learningRate(0.01)// .biasLearningRate(0.02)
				// .learningRateDecayPolicy(LearningRatePolicy.Inverse).lrPolicyDecayRate(0.001).lrPolicyPower(0.75)
				.weightInit(WeightInit.XAVIER).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.updater(Updater.NESTEROVS).momentum(0.9).list()
				.layer(0,
						new ConvolutionLayer.Builder(5, 5)
								// nIn and nOut specify depth. nIn here is the
								// nChannels and nOut is the number of filters
								// to be applied
								.nIn(nChannels).stride(1, 1).nOut(20).activation("identity").build())
				.layer(1,
						new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2)
								.build())
				.layer(2,
						new ConvolutionLayer.Builder(5, 5)
								// Note that nIn need not be specified in later
								// layers
								.stride(1, 1).nOut(50).activation("identity").build())
				.layer(3,
						new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2)
								.build())
				.layer(4, new DenseLayer.Builder().activation("relu").nOut(500).build())
				.layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nOut(outputNum)
						.activation("softmax").build())
				.backprop(true).pretrain(false);
		// The builder needs the dimensions of the image along with the number
		// of channels. these are 28x28 images in one channel
		new ConvolutionLayerSetup(builder, 100, 100, 1);

		MultiLayerConfiguration conf = builder.build();
		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();

		log.info("Train model....");
		model.setListeners(new ScoreIterationListener(1));
		for (int i = 0; i < nEpochs; i++) {
			model.fit(trainIterator);
			log.info("*** Completed epoch {} ***", i);

			log.info("Evaluate model....");
			Evaluation eval = new Evaluation(outputNum);
			while (testIterator.hasNext()) {
				DataSet ds = testIterator.next();
				INDArray output = model.output(ds.getFeatureMatrix(), false);
				eval.eval(ds.getLabels(), output);
			}
			log.info(eval.stats());
			testIterator.reset();

			System.out.println("Do you want to keep trainnig? Y(Yes) N(No)");
			String answer = scanner.next();

			if ("n".equalsIgnoreCase(answer.trim())) {
				break;
			}
		}

		log.info("Saving the model {}", outputFile);
		ModelSerializer.writeModel(model, new File(outputFile), true);

	}

	public List<Tweet> readFile(String path) {
		List<Tweet> lines = new ArrayList<>();
		BufferedReader file = null;
		try {
			file = new BufferedReader(new FileReader(path));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			throw new RuntimeException(e);
		}
		try {
			while (file.ready()) {
				String line = file.readLine();
				String[] tokens = line.split("\t");
				Tweet tweet = parseTweet(tokens[1]);
				tweet.setLabel(parseLabel(tokens[0]));
				lines.add(tweet);
			}
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
		return lines;
	}

	private Tweet parseTweet(String line) {
		Tweet tweet = new Tweet();

		String[] tokens = line.split(" ");
		INDArray array = null; // max size

		int cont = 0;
		for (String token : tokens) {
			if (cont >= 100)
				break;
			INDArray features = lookuptable.vector(token.toLowerCase());

			if (features != null) {
				if (array == null) {
					array = features;
				} else {
					array = Nd4j.concat(1, array, features);
				}
			} else {
				if (array == null)
					array = Nd4j.zeros(1, 100);
				else
					array = Nd4j.concat(1, array, Nd4j.zeros(1, 100));
			}

			cont++;
		}

		if (cont < 100) {

			for (int i = 0; i < 100 - cont; i++) {

				array = Nd4j.concat(1, array, Nd4j.zeros(1, 100));

			}

		}

		tweet.setFeatures(array);
		return tweet;
	}

	private int parseLabel(String line) {
		if (line.equals("positive")) {
			return 1;
		}
		return 0;
	}

}
