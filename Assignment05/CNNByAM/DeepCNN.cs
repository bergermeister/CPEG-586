using CNNByAM;
using MatrixLibByAM;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNNAM
{
    [Serializable]
    class DeepCNN
    { // assembles the entire deep CNN network
        public DeepCNN(List<CNNLayer> CNNLayerList, List<Layer> layerList,
            List<Matrix> inputDataList, List<Matrix> outputLabels, int batchSize)
        {
            this.CNNLayerList = CNNLayerList;
            this.LayerList = layerList;
            this.InputDataList = inputDataList;
            this.OutputLabels = outputLabels;
            this.batchSize = batchSize;
            Flatten = new Matrix[batchSize];
        }
        int batchSize = 0;
        public List<CNNLayer> CNNLayerList = null;
        public List<Layer> LayerList = new List<Layer>();
        List<Matrix> InputDataList = null;
        List<Matrix> OutputLabels = null;
        Matrix[] Flatten = null;

        public Matrix Evaluate(Matrix inputData, int batchIndex)
        {
            for (int i = 0; i < CNNLayerList.Count; i++)
            {
                List<Matrix> PrevLayerOutputList = new List<Matrix>();
                if (i == 0)
                    PrevLayerOutputList.Add(inputData);
                else
                {
                    PrevLayerOutputList.Clear();
                    foreach (FeatureMap fmp in CNNLayerList[i - 1].FeatureMapList)
                    {
                        PrevLayerOutputList.Add(fmp.OutPutSS[batchIndex]);
                    }
                }
                CNNLayerList[i].Evaluate(PrevLayerOutputList,batchIndex);
            }
            // flatten each feature map in the CNN layer and assemble
            // all maps into an nx1 vector
            int outputSSsize = CNNLayerList[CNNLayerList.Count - 1].FeatureMapList[0].OutPutSS[batchIndex].Rows;
            int flattenSize = outputSSsize * outputSSsize  
                * CNNLayerList[CNNLayerList.Count - 1].FeatureMapList.Count;
            Flatten[batchIndex] = new Matrix(flattenSize, 1);
            int index = 0;
            foreach (FeatureMap fmp in CNNLayerList[CNNLayerList.Count - 1].FeatureMapList)
            {
                Matrix ss = fmp.OutPutSS[batchIndex].Flatten();  // output of a feature map
                for (int i = 0; i < ss.Rows; i++)
                {
                    Flatten[batchIndex].D[index][0] = ss.D[i][0];
                    index++;
                }
            }
            int count = 0;
            Matrix res = null;
            foreach (Layer layer in LayerList)  // regular NN Layers
            {
                if (count == 0)
                    res = layer.Evaluate(Flatten[batchIndex],batchIndex,false);  // first layer
                else
                    res = layer.Evaluate(res,batchIndex,false);
                count++;
            }
            return res;
        }

        public void Train(int numEpochs, double learningRate, int batchSize)
        {
            double trainingError = 0;
            object olock = new object();
            
            for (int i = 0; i < numEpochs; i++)
            {
                trainingError = 0;
                RandomizeInputs();
                int dj = 0; // data index
                for (int j = 0; j < InputDataList.Count / batchSize; j++)
                {
                    dj = j * batchSize;
                    //Stopwatch sw = new Stopwatch();
                    //sw.Start();
                    double[] trainingErr = new double[batchSize];
                    Parallel.For(0, batchSize, (b) =>
                    //for (int b = 0; b < batchSize; b++)
                    {
                        // do forward pass
                        Matrix res = this.Evaluate(InputDataList[dj + b], b);

                        // compute training error
                        for (int h = 0; h < res.Rows; h++)
                            trainingErr[b] += (res.D[h][0] - OutputLabels[dj + b].D[h][0])
                                * (res.D[h][0] - OutputLabels[dj + b].D[h][0]);

                        // ----------compute deltas on regular NN layers--------------
                        for (int count = LayerList.Count - 1; count >= 0; count--)
                        {
                            var layer = LayerList[count];
                            if (count == (LayerList.Count - 1))  // last layer
                            {
                                layer.Delta[b] = (OutputLabels[dj + b] - layer.A[b]).Mul(-1);  // for softmax by default
                                if (layer.activationType == ActivationType.SIGMOID)
                                    layer.Delta[b] = layer.Delta[b].ElementByElementMul(layer.APrime[b]);
                                if (layer.activationType == ActivationType.RELU)
                                {
                                    for (int m = 0; m < layer.numNeurons; m++)
                                    {
                                        if (layer.Sum[b].D[m][0] < 0)
                                            layer.Delta[b].D[m][0] = 0;
                                    }
                                }
                            }
                            else  // previous layer
                            {
                                layer.Delta[b] = LayerList[count + 1].W.Transpose() * LayerList[count + 1].Delta[b];
                                //-------apply dropout----------------------
                                if (layer.Dropout < 1.0)
                                {
                                    layer.Delta[b] = layer.Delta[b].ElementByElementMul(layer.DropM[b]);
                                }
                                //--------------------------------------------

                                if (layer.activationType == ActivationType.SIGMOID)
                                    layer.Delta[b] = layer.Delta[b].ElementByElementMul(layer.APrime[b]);
                                if (layer.activationType == ActivationType.RELU)
                                {
                                    for (int m = 0; m < layer.numNeurons; m++)
                                    {
                                        if (layer.Sum[b].D[m][0] < 0)
                                            layer.Delta[b].D[m][0] = 0;
                                    }
                                }
                            }
                            layer.GradB[b] = layer.GradB[b] + layer.Delta[b];
                            if (count == 0)  // first NN layer connected to CNN last layer via Flatten
                                layer.GradW[b] = layer.GradW[b] + (layer.Delta[b] * Flatten[b].Transpose()); // flatten = previous output
                            else
                                layer.GradW[b] = layer.GradW[b] + (layer.Delta[b] * LayerList[count - 1].A[b].Transpose());
                        }
                        // compute delta on the output of SS (flat) layer of all feature maps
                        Matrix deltaSSFlat = this.LayerList[0].W.Transpose() * this.LayerList[0].Delta[b];

                        // do reverse flattening and distribute the deltas on
                        // each feature map's SS (SubSampling layer)
                        int index = 0;
                        // last CNN layer
                        foreach (FeatureMap fmp in CNNLayerList[CNNLayerList.Count - 1].FeatureMapList)
                        {
                            fmp.DeltaSS[b] = new Matrix(fmp.OutPutSS[b].Rows, fmp.OutPutSS[b].Cols);
                            for (int m = 0; m < fmp.OutPutSS[b].Rows; m++)
                            {
                                for (int n = 0; n < fmp.OutPutSS[b].Cols; n++)
                                {
                                    fmp.DeltaSS[b].D[m][n] = deltaSSFlat.D[index][0];
                                    index++;
                                }
                            }
                        }
                        // process CNN layers in reverse order, from last layer towards input
                        for (int cnnCount = CNNLayerList.Count - 1; cnnCount >= 0; cnnCount--)
                        {
                            // compute deltas on the C layers - distrbute deltas from SS layer
                            // then multiply by the activation function
                            //foreach (FeatureMap fmp in CNNLayerList[cnnCount].FeatureMapList)
                            //Parallel.For(0, CNNLayerList[cnnCount].FeatureMapList.Count, (k) =>
                            for (int k = 0; k < CNNLayerList[cnnCount].FeatureMapList.Count; k++)
                            {
                                FeatureMap fmp = CNNLayerList[cnnCount].FeatureMapList[k];
                                int indexm = 0; int indexn = 0;
                                fmp.DeltaCV[b] = new Matrix(fmp.OutPutSS[b].Rows * 2, fmp.OutPutSS[b].Cols * 2);
                                for (int m = 0; m < fmp.DeltaSS[b].Rows; m++)
                                {
                                    indexn = 0;
                                    for (int n = 0; n < fmp.DeltaSS[b].Cols; n++)
                                    {
                                        if (fmp.activationType == ActivationType.SIGMOID)
                                        {
                                            fmp.DeltaCV[b].D[indexm][indexn] = (1 / 4.0) * fmp.DeltaSS[b].D[m][n] * fmp.APrime[b].D[indexm][indexn];
                                            fmp.DeltaCV[b].D[indexm][indexn + 1] = (1 / 4.0) * fmp.DeltaSS[b].D[m][n] * fmp.APrime[b].D[indexm][indexn + 1];
                                            fmp.DeltaCV[b].D[indexm + 1][indexn] = (1 / 4.0) * fmp.DeltaSS[b].D[m][n] * fmp.APrime[b].D[indexm + 1][indexn];
                                            fmp.DeltaCV[b].D[indexm + 1][indexn + 1] = (1 / 4.0) * fmp.DeltaSS[b].D[m][n] * fmp.APrime[b].D[indexm + 1][indexn + 1];
                                            indexn = indexn + 2;
                                        }
                                        if (fmp.activationType == ActivationType.RELU)
                                        {
                                            if (fmp.Sum[b].D[indexm][indexn] > 0)
                                                fmp.DeltaCV[b].D[indexm][indexn] = (1 / 4.0) * fmp.DeltaSS[b].D[m][n];
                                            else
                                                fmp.DeltaCV[b].D[indexm][indexn] = 0;
                                            if (fmp.Sum[b].D[indexm][indexn + 1] > 0)
                                                fmp.DeltaCV[b].D[indexm][indexn + 1] = (1 / 4.0) * fmp.DeltaSS[b].D[m][n];
                                            else
                                                fmp.DeltaCV[b].D[indexm][indexn + 1] = 0;
                                            if (fmp.DeltaCV[b].D[indexm + 1][indexn] > 0)
                                                fmp.DeltaCV[b].D[indexm + 1][indexn] = (1 / 4.0) * fmp.DeltaSS[b].D[m][n];
                                            else
                                                fmp.DeltaCV[b].D[indexm + 1][indexn] = 0;
                                            if (fmp.DeltaCV[b].D[indexm + 1][indexn + 1] > 0)
                                                fmp.DeltaCV[b].D[indexm + 1][indexn + 1] = (1 / 4.0) * fmp.DeltaSS[b].D[m][n];
                                            else
                                                fmp.DeltaCV[b].D[indexm + 1][indexn + 1] = 0;
                                            indexn = indexn + 2;
                                        }
                                    }
                                    indexm = indexm + 2;
                                }
                            }

                            //----------compute BiasGrad in current CNN Layer-------
                            foreach (FeatureMap fmp in CNNLayerList[cnnCount].FeatureMapList)
                            {
                                for (int u = 0; u < fmp.DeltaCV[b].Rows; u++)
                                {
                                    for (int v = 0; v < fmp.DeltaCV[b].Cols; v++)
                                        lock (olock)
                                        {
                                            fmp.BiasGrad += fmp.DeltaCV[b].D[u][v];
                                        }
                                }
                            }
                            //----------compute gradients for pxq kernels in current CNN layer--------
                            if (cnnCount > 0)  // not the first CNN layer
                            {
                                for (int p = 0; p < CNNLayerList[cnnCount - 1].FeatureMapList.Count; p++)
                                //Parallel.For(0, CNNLayerList[cnnCount - 1].FeatureMapList.Count, (p) =>
                                {
                                    for (int q = 0; q < CNNLayerList[cnnCount].FeatureMapList.Count; q++)
                                    {
                                        lock (olock)
                                        {
                                            CNNLayerList[cnnCount].KernelGrads[p, q] = CNNLayerList[cnnCount].KernelGrads[p, q] +
                                                CNNLayerList[cnnCount - 1].FeatureMapList[p].OutPutSS[b].RotateBy90().RotateBy90().Convolution(CNNLayerList[cnnCount].FeatureMapList[q].DeltaCV[b]);
                                        }
                                    }
                                }
                                //---------------this layer is done, now backpropagate to prev CNN Layer----------
                                for (int p = 0; p < CNNLayerList[cnnCount - 1].FeatureMapList.Count; p++)
                                //Parallel.For(0, CNNLayerList[cnnCount - 1].FeatureMapList.Count, (p) =>
                                {
                                    int size = CNNLayerList[cnnCount - 1].FeatureMapList[p].OutPutSS[b].Rows;
                                    CNNLayerList[cnnCount - 1].FeatureMapList[p].DeltaSS[b] = new Matrix(size, size);
                                    //CNNLayerList[cnnCount - 1].FeatureMap2List[p].DeltaSS.Clear();
                                    for (int q = 0; q < CNNLayerList[cnnCount].FeatureMapList.Count; q++)
                                    {
                                        CNNLayerList[cnnCount - 1].FeatureMapList[p].DeltaSS[b] = CNNLayerList[cnnCount - 1].FeatureMapList[p].DeltaSS[b] +
                                        CNNLayerList[cnnCount].FeatureMapList[q].DeltaCV[b].ConvolutionFull(
                                        CNNLayerList[cnnCount].Kernels[p, q].RotateBy90().RotateBy90());
                                    }
                                }
                            }
                            else  // very first CNN layer which is connected to input
                            {     // has 1xnumFeaturemaps 2-D array of Kernels and Kernel Gradients
                                //----------compute gradient for first layer cnn kernels--------
                                for (int p = 0; p < 1; p++)
                                {
                                    for (int q = 0; q < CNNLayerList[cnnCount].FeatureMapList.Count; q++)
                                    {
                                        lock (olock)
                                        {
                                            CNNLayerList[cnnCount].KernelGrads[p, q] = CNNLayerList[cnnCount].KernelGrads[p, q] +
                                                InputDataList[dj + b].RotateBy90().RotateBy90().Convolution(CNNLayerList[cnnCount].FeatureMapList[q].DeltaCV[b]);
                                        }
                                    }
                                }
                            }
                        }
                    }); // end parallel b loop
                    //sw.Stop();
                    //Console.WriteLine("Time per batch = " + sw.ElapsedMilliseconds);
                    //Stopwatch sw = new Stopwatch();
                    //sw.Start();
                    trainingError += trainingErr.Sum();
                    UpdateKernelsWeightsBiases(learningRate, batchSize);
                    ClearGradients();
                    //sw.Stop();
                    //Console.WriteLine("Time per batch = " + sw.ElapsedMilliseconds);
                }

                if (i % 10 == 0)
                    learningRate = learningRate / 2;  // reduce learning rate

                Console.WriteLine("epoch = " + i.ToString() + " training error = " + trainingError.ToString());
            }
        }

        void ClearGradients()
        {
            for (int cnnCount = 0; cnnCount < CNNLayerList.Count; cnnCount++)
            {
                if (cnnCount == 0)  // first CNN layer
                {
                    for (int p = 0; p < 1; p++)
                    {
                        for (int q = 0; q < CNNLayerList[cnnCount].FeatureMapList.Count; q++)
                        {
                            CNNLayerList[cnnCount].KernelGrads[p, q].Clear();
                        }
                    }
                }
                else  // next CNN layers
                {
                    for (int p = 0; p < CNNLayerList[cnnCount - 1].FeatureMapList.Count; p++)
                    {
                        for (int q = 0; q < CNNLayerList[cnnCount].FeatureMapList.Count; q++)
                        {
                            CNNLayerList[cnnCount].KernelGrads[p, q].Clear();
                        }
                    }
                }
                foreach (FeatureMap fmp in CNNLayerList[cnnCount].FeatureMapList)
                {
                    fmp.BiasGrad = 0;
                }
            }
            foreach (Layer layer in LayerList)
            {
                for (int b = 0; b < layer.batchSize; b++)
                {
                    layer.GradW[b].Clear();
                    layer.GradB[b].Clear();
                }
            }
        }

        void UpdateKernelsWeightsBiases(double learningRate, int batchSize)
        {
            //---------------update kernels and weights-----
            for (int cnnCount = 0; cnnCount < CNNLayerList.Count; cnnCount++)
            //Parallel.For (0,CNNLayerList.Count, (cnnCount) =>
            {
                if (cnnCount == 0)  // first CNN layer
                {
                    for (int p = 0; p < 1; p++)
                    {
                        for (int q = 0; q < CNNLayerList[0].FeatureMapList.Count; q++)
                        {
                            CNNLayerList[cnnCount].Kernels[p, q] = CNNLayerList[cnnCount].Kernels[p, q] -
                                CNNLayerList[cnnCount].KernelGrads[p, q].Mul(1.0 / batchSize).Mul(learningRate);
                        }
                    }
                }
                else  // next CNN layers
                {
                    for (int p = 0; p < CNNLayerList[cnnCount-1].FeatureMapList.Count; p++)
                    {
                        for (int q = 0; q < CNNLayerList[cnnCount].FeatureMapList.Count; q++)
                        {
                            CNNLayerList[cnnCount].Kernels[p, q] = CNNLayerList[cnnCount].Kernels[p, q] -
                                CNNLayerList[cnnCount].KernelGrads[p, q].Mul(1.0 / batchSize).Mul(learningRate);
                        }
                    }
                }
                foreach (FeatureMap fmp in CNNLayerList[cnnCount].FeatureMapList)
                {
                    //fmp.Kernel = fmp.Kernel - fmp.KernelGrad.RotateBy90().RotateBy90().Mul(1.0 / batchSize).Mul(learningRate);
                    fmp.Bias = fmp.Bias - (fmp.BiasGrad / batchSize) * learningRate;
                }
            }

            //foreach (Layer layer in LayerList)
            //{
            //    layer.W = layer.W - layer.GradW.Mul(1.0 / batchSize).Mul(learningRate);
            //    layer.B = layer.B - layer.GradB.Mul(1.0 / batchSize).Mul(learningRate);
            //}
            foreach (Layer layer in LayerList)
            {
                Matrix GradW = new Matrix(layer.GradW[0].Rows, layer.GradW[0].Cols);
                for (int b = 0; b < batchSize; b++)
                {
                    GradW = GradW + layer.GradW[b];
                }
                Matrix GradB = new Matrix(layer.GradB[0].Rows, layer.GradB[0].Cols);
                for (int b = 0; b < batchSize; b++)
                {
                    //GradB = GradB + layer.GradW[b];
                    GradB = GradB + layer.GradB[b];
                }
                layer.W = layer.W - GradW.Mul(1.0 / batchSize*learningRate);
                layer.B = layer.B - GradB.Mul(1.0 / batchSize*learningRate);
            }
        }

        void RandomizeInputs()
        {
            // randomize input data order for stochastic behavior
            Random rand = new Random();
            for (int i = 0; i < InputDataList.Count; i++)
            {
                int num1 = rand.Next(InputDataList.Count);
                int num2 = rand.Next(InputDataList.Count);
                // exhange inputdata and output labels
                var tempInputData = InputDataList[num2].Clone();
                InputDataList[num2] = InputDataList[num1];
                InputDataList[num1] = tempInputData;
                var tempLabel = OutputLabels[num2].Clone();
                OutputLabels[num2] = OutputLabels[num1];
                OutputLabels[num1] = tempLabel;
            }
        }
    }
}
