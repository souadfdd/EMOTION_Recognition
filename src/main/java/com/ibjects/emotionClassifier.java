//import org.apache.log4j.BasicConfigurator;
package com.ibjects;
import org.apache.log4j.BasicConfigurator;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.Random;

public class emotionClassifier {
    public static final String path = "C:\\Users\\pc\\IdeaProjects\\emotion\\src\\main\\resources\\FER_modified";
    private static final int HEIGHT = 48;
    private static final int WIDTH = 48;
    private static final int DEPTH = 1;
    private static final int N_OUTCOMES = 7;

    public static void main(String[] args) throws IOException, InterruptedException {
        int seed = 1234;
        int nEpochs = 50;
        BasicConfigurator.configure();

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().
                seed(seed)
                .updater(new Adam(0.0005))
                .list()
                .setInputType(InputType.convolutionalFlat(48, 48, 1))
                .layer(0, new ConvolutionLayer.Builder().nIn(1).nOut(20).activation(Activation.RELU).kernelSize(5, 5).stride(1, 1).build())
                .layer(1, new SubsamplingLayer.Builder().kernelSize(2, 2).stride(2, 2).poolingType(SubsamplingLayer.PoolingType.MAX).build())
                .layer(2, new ConvolutionLayer.Builder().nOut(50).activation(Activation.RELU).kernelSize(5, 5).stride(1, 1).build())
                .layer(3, new SubsamplingLayer.Builder().kernelSize(2, 2).stride(2, 2).poolingType(SubsamplingLayer.PoolingType.MAX).build())
                .layer(4, new DenseLayer.Builder().nOut(500).activation(Activation.RELU).build())
                .layer(5, new OutputLayer.Builder().nOut(N_OUTCOMES).activation(Activation.SOFTMAX).lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        File fileTrain = new File(path + "/train");
        FileSplit fileSplitTrain = new FileSplit(fileTrain, NativeImageLoader.ALLOWED_FORMATS, new Random(seed));
        RecordReader recordReaderTrain = new ImageRecordReader(HEIGHT, WIDTH, DEPTH, new ParentPathLabelGenerator());
        recordReaderTrain.initialize(fileSplitTrain);
        int BatcheSize = 50;
        DataSetIterator DataSetIteratorTrain = new RecordReaderDataSetIterator(recordReaderTrain, BatcheSize, 1, N_OUTCOMES);
        DataNormalization scalar = new ImagePreProcessingScaler(0, 1);
        DataSetIteratorTrain.setPreProcessor(scalar);
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        model.setListeners(new StatsListener(statsStorage));

        for (int i = 0; i < nEpochs; i++) {
            model.fit(DataSetIteratorTrain);
        }
        File fileTest = new File(path + "/test");
        FileSplit fileSplitTest = new FileSplit(fileTest, NativeImageLoader.ALLOWED_FORMATS, new Random(seed));
        RecordReader recordReaderTest = new ImageRecordReader(HEIGHT, WIDTH, DEPTH, new ParentPathLabelGenerator());
        recordReaderTest.initialize(fileSplitTest);
        DataSetIterator DataSetIteratorTest = new RecordReaderDataSetIterator(recordReaderTest, BatcheSize, 1, N_OUTCOMES);
        DataNormalization scalartest = new ImagePreProcessingScaler(0, 1);
        DataSetIteratorTrain.setPreProcessor(scalartest);
        Evaluation evaluation = new Evaluation();
        while (DataSetIteratorTest.hasNext()) {
            DataSet dataset = DataSetIteratorTest.next();
            INDArray features = dataset.getFeatures();
            INDArray targetlabels = dataset.getLabels();
            INDArray predicted = model.output(features);

            evaluation.eval(predicted, targetlabels);
        }
        System.out.print(evaluation.stats());
        ModelSerializer.writeModel(model, new File("model.zip"), true);
    }
}