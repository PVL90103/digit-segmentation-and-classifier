package digitrecognition;

import java.io.File;
import java.util.Random;

//import mnistdatareader.MnistDataReader;
//import mnistdatareader.MnistMatrix;
import org.apache.log4j.BasicConfigurator;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.deeplearning4j.datasets.iterator.impl.EmnistDataSetIterator;

import org.deeplearning4j.util.ModelSerializer;
/**
 * I used an example from deeplearning4j to build this example (original autho agibsonccc)
 * https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/convolution/LenetMnistExample.java
 * @author Pavel
 */
public class NetworkTrainer {
    private static final Logger log = LoggerFactory.getLogger(NetworkTrainer.class);


    public static void main(String[] args) throws Exception {

        BasicConfigurator.configure();

        int height = 28;    // height of the picture in px
        int width = 28;     // width of the picture in px
        int channels = 1;   // single channel for grayscale images
        int outputNum = 10; // The number of possible outcomes
        int batchSize = 64; // Test batch size
        int nEpochs = 5; // Number of training epochs
        int iterations = 1; // Number of training iterations
        int seed = 123; // number used to initialize a pseudorandom number generator.

        Random randNumGen = new Random(seed);

        log.info("Load data....");


        log.info("Data vectorization...");
        // vectorization of train data
        File trainData = new File("data/mnist_png/training");
        FileSplit trainSplit = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator(); // use parent directory name as the image label
        ImageRecordReader trainRR = new ImageRecordReader(height, width, channels, labelMaker);
        trainRR.initialize(trainSplit);
        DataSetIterator trainIter = new RecordReaderDataSetIterator(trainRR, batchSize, 1, outputNum);

        // pixel values from 0-255 to 0-1 (min-max scaling)
        DataNormalization imageScaler = new ImagePreProcessingScaler();
        imageScaler.fit(trainIter);
        trainIter.setPreProcessor(imageScaler);

        // vectorization of test data
        File testData = new File("data/mnist_png/testing");
        FileSplit testSplit = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
        ImageRecordReader testRR = new ImageRecordReader(height, width, channels, labelMaker);
        testRR.initialize(testSplit);
        DataSetIterator testIter = new RecordReaderDataSetIterator(testRR, batchSize, 1, outputNum);
        testIter.setPreProcessor(imageScaler); // same normalization for better results

        log.info("Build model....");


//        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
//                .seed(seed)
//                .iterations(iterations)
//                .regularization(true).l2(0.0005)
//                .learningRate(.01)
//                .weightInit(WeightInit.XAVIER)
//                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//                .updater(Updater.NESTEROVS)
//                //.updater(Updater.ADAM)
//                //.updater(Updater.ADADELTA)
//                .list()
//                .layer(0, new ConvolutionLayer.Builder(5, 5)
//                        .nIn(channels)
//                        .stride(1, 1)
//                        .nOut(20)
//                        .activation(Activation.IDENTITY)
//                        .build())
//                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
//                        .kernelSize(2,2)
//                        .stride(2,2)
//                        .build())
//                .layer(2, new ConvolutionLayer.Builder(5, 5)
//                        .stride(1, 1)
//                        .nOut(50)
//                        .activation(Activation.IDENTITY)
//                        .build())
//                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
//                        .kernelSize(2,2)
//                        .stride(2,2)
//                        .build())
//                .layer(4, new DenseLayer.Builder().activation(Activation.RELU)
//                        .nOut(500).build())
//                .layer(5,
//                        new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
//                                .nOut(outputNum)
//                                .activation(Activation.SOFTMAX)
//                                .build())
//                .setInputType(InputType.convolutionalFlat(28,28,1))
//                .backprop(true).pretrain(false).build();
//
//
//        MultiLayerNetwork net = new MultiLayerNetwork(conf);
//        net.init();
//        net.setListeners(new ScoreIterationListener(10));
//        log.info("Total num of params: {}", net.numParams());
//
//        // evaluation while training (the score should go down)
//        for (int i = 0; i < nEpochs; i++) {
//            net.fit(trainIter);
//            log.info("Completed epoch {}", i);
//            Evaluation eval = net.evaluate(testIter);
//            log.info(eval.stats());
//
//            trainIter.reset();
//            testIter.reset();
//        }
//
//        File ministModelPath = new File("/minist-model.zip");
//        ModelSerializer.writeModel(net, ministModelPath, true);
//        log.info("The MINIST model has been saved in {}", ministModelPath.getPath());

    }

}