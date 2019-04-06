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
    class Network
    { // contains list of layers for a regular NN
        List<Layer> LayerList = new List<Layer>();
        public Network(List<Layer> layerList, List<Matrix> inputDataList, List<Matrix> outputLabels)
        {
            this.LayerList = layerList;
            this.InputDataList = inputDataList;
            this.OutputLabels = outputLabels;
        }
        List<Matrix> InputDataList = null;
        List<Matrix> OutputLabels = null;
        public Matrix Evaluate(Matrix inputData, int batchIndex, bool doBatchNorm = false, bool doBNTestMode = false)
        {
            int count = 0;
            Matrix res = null;
            foreach (Layer layer in LayerList)  // regular NN Layers
            {
                if (count == 0)
                    res = layer.Evaluate(inputData, batchIndex, doBatchNorm, doBNTestMode);  // first layer
                else
                {
                    if (count == LayerList.Count - 1)  // last layer - no batch norm
                        res = layer.Evaluate(res, batchIndex, false, false);
                    else
                        res = layer.Evaluate(res, batchIndex, doBatchNorm, doBNTestMode);
                }
                count++;
            }
            return res;
        }

        public void Train(int numEpochs, double learningRate, int batchSize, bool doBatchNorm = false)
        {
            double trainingError = 0;
            Stopwatch sw = new Stopwatch();
            object olock = new object();
            for (int i = 0; i < numEpochs; i++)
            {
                trainingError = 0;
               // int bi = 0; // batch index
                int dj = 0; // data index
                for (int j = 0; j < InputDataList.Count / batchSize; j++)
                {
                    dj = j * batchSize;
                    //for (int b = 0; b < batchSize; b++)  // can be done in parallel
                    sw.Start();
                    ClearGradients();
                    //--------------------------batch norm extra step------------
                    if (doBatchNorm == true)
                    {
                        for (int b = 0; b < batchSize; b++)
                        //Parallel.For(0, batchSize, (b) =>
                        {
                            // do forward pass - just to compute batch mean and var
                            lock (olock)
                            {
                                Matrix res = this.Evaluate(InputDataList[dj + b].Flatten(), b, false,false);
                            }
                        }//);
                        for (int count = LayerList.Count - 1; count >= 0; count--)
                        {
                            if (count != (LayerList.Count - 1))  // skip last layer for batch mean
                            {
                                lock (olock)
                                {
                                    var layer = LayerList[count];
                                    for (int bb = 0; bb < batchSize; bb++)
                                    {
                                        layer.Mu = layer.Mu + layer.Sum[bb];
                                    }
                                    layer.Mu = (layer.Mu.Mul(1.0 / batchSize));
                                    // now compute variance
                                    for (int bb = 0; bb < batchSize; bb++)
                                    {
                                        layer.Var = layer.Var + (layer.Sum[bb] - layer.Mu).ElementByElementMul
                                           (layer.Sum[bb] - layer.Mu);
                                    }
                                    layer.Var = layer.Var.Mul(1.0 / batchSize);
                                    for (int kk = 0; kk < layer.Var.Rows; kk++)
                                        layer.Ivar.D[kk][0] = 1.0 / Math.Sqrt(layer.Var.D[kk][0] + layer.epsilon);
                                    
                                    // running_mean = momentum * running_mean + (1 - momentum) * batch_mean
                                    // running_var = momentum * running_var + (1 - momentum) * batch_var
                                    layer.RunningMu = layer.RunningMu.Mul(layer.MomentumBN) +
                                        layer.Mu.Mul(1 - layer.MomentumBN);
                                    layer.RunningVar = layer.RunningVar.Mul(layer.MomentumBN) +
                                        layer.Var.Mul(1 - layer.MomentumBN);
                                }
                            }
                        }
                    }
                    //---------------------end batch norm extra step-------------------
                    for (int b = 0; b < batchSize; b++)
                    //Parallel.For(0, batchSize, (b) =>
                    {
                        // do forward pass
                        Matrix res = this.Evaluate(InputDataList[dj + b].Flatten(), b,doBatchNorm, false);
                        lock (olock)
                        {
                            for (int h = 0; h < res.Rows; h++)
                                trainingError += (res.D[h][0] - OutputLabels[dj + b].D[h][0]) * (res.D[h][0] - OutputLabels[dj + b].D[h][0]);
                        }
                        // ----------compute deltas on NN layers--------------
                        for (int count = LayerList.Count - 1; count >= 0; count--)
                        {
                            var layer = LayerList[count];
                            if (count == (LayerList.Count - 1))  // last layer
                            {
                                layer.Delta[b] = (OutputLabels[dj + b] - layer.A[b]).Mul(-1);  // for softmax by default
                                if (layer.activationType == ActivationType.SIGMOID)
                                    //layer.Delta[b] = layer.Delta[b].ElementByElementMul(layer.APrime[b]);
                                    layer.Delta[b] = layer.Delta[b].ElementByElementMul(layer.A[b].ElementByElementMul(layer.A[b].Mul(-1).Add(1)));
                                if (layer.activationType == ActivationType.RELU)
                                {
                                    for (int m = 0; m < layer.numNeurons; m++)
                                    {
                                        if (layer.Sum[b].D[m][0] < 0)
                                            layer.Delta[b].D[m][0] = 0;
                                    }
                                }
                                layer.GradB[b] = layer.Delta[b];
                                layer.GradW[b] = (layer.Delta[b] * LayerList[count - 1].A[b].Transpose());
                            }
                            else
                            {  // previous layer (not the last layer)
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
                                //if (doBatchNorm == true)
                                //{
                                //    //---------batch norm back prop---------------------
                                //    // dbeta = np.sum(dout, axis=0)
                                //    // dgamma = np.sum(xhat * dout, axis = 0)
                                //    // dx = (gamma * istd / N) * (N * dout - xhat * dgamma - dbeta)
                                //    lock (olock)
                                //    {
                                //        for (int kk = 0; kk < layer.dBeta.Rows; kk++)
                                //            layer.dBeta.D[kk][0] += layer.Delta[b].D[kk][0];
                                //        for (int kk = 0; kk < layer.dBeta.Rows; kk++)
                                //            layer.dGamma.D[kk][0] += layer.Xhat[b].D[kk][0] * layer.Delta[b].D[kk][0];
                                //        layer.Delta[b] = layer.Ivar.ElementByElementMul(layer.Gamma).Mul(1.0 / batchSize).ElementByElementMul
                                //        (layer.Delta[b].Mul(batchSize) - layer.Xhat[b].ElementByElementMul(layer.dGamma) - layer.dBeta);
                                //    }
                                //}
                            }

                            //layer.GradB[b] = layer.Delta[b];
                            if (count == 0)  // first layer connected to input
                            {
                                //layer.GradW[b] = (layer.Delta[b] * InputDataList[dj + b].Flatten().Transpose()); // flatten = previous output
                            }
                            else
                                layer.GradW[b] = (layer.Delta[b] * LayerList[count - 1].A[b].Transpose());
                        }
                    }//); // end of batch
                    sw.Stop();
                    if (doBatchNorm == true)
                    {
                        //---------batch norm back prop---------------------
                        // dbeta = np.sum(dout, axis=0)
                        // dgamma = np.sum(xhat * dout, axis = 0)
                        // dx = (gamma * istd / N) * (N * dout - xhat * dgamma - dbeta)
                        lock (olock)
                        {
                            for (int count = LayerList.Count - 1; count >= 0; count--)
                            {
                                if (count != (LayerList.Count - 1))
                                {
                                    for (int kk = 0; kk < LayerList[count].dBeta.Rows; kk++)
                                    {
                                        for (int b = 0; b < batchSize; b++)
                                        {
                                            LayerList[count].dBeta.D[kk][0] += LayerList[count].Delta[b].D[kk][0];
                                        }
                                    }

                                    for (int kk = 0; kk < LayerList[count].dGamma.Rows; kk++)
                                    {
                                        for (int b = 0; b < batchSize; b++)
                                        {
                                            LayerList[count].dGamma.D[kk][0] += LayerList[count].Xhat[b].D[kk][0]
                                                * LayerList[count].Delta[b].D[kk][0];
                                        }
                                    }
                                    for (int b = 0; b < batchSize; b++)
                                    { // dx = (gamma * istd / N) * (N * dout - xhat * dgamma - dbeta)
                                        LayerList[count].Delta[b] = LayerList[count].Ivar.ElementByElementMul
                                            (LayerList[count].Gamma).Mul(1.0 / batchSize).ElementByElementMul
                                            (LayerList[count].Delta[b].Mul(batchSize) 
                                            - LayerList[count].Xhat[b].ElementByElementMul(LayerList[count].dGamma) 
                                            - LayerList[count].dBeta);
                                    }
                                    for (int b = 0; b < batchSize; b++)
                                    {
                                        LayerList[count].GradB[b] = LayerList[count].Delta[b];
                                        if (count == 0)
                                            LayerList[count].GradW[b] = (LayerList[count].Delta[b] * InputDataList[dj + b].Flatten().Transpose()); // flatten = previous output
                                    //    else
                                    //        LayerList[count].GradW[b] = (LayerList[count].Delta[b] * LayerList[count - 1].A[b].Transpose());
                                    }
                                }
                            }
                        }
                    }
                    //Console.WriteLine("Time per batch = " + sw.ElapsedMilliseconds);
                    UpdateKernelsWeightsBiases(learningRate, batchSize);
                    ClearGradients();
                }
                //if ((i % 10) == 0)
                //    learningRate = learningRate - learningRate/2.0;

                Console.WriteLine("epoch = " + i.ToString() + " training error = " + trainingError.ToString());
            }
        }
        void ClearGradients()
        {
            foreach (Layer layer in LayerList)
            {
                //for (int i = 0; i < layer.numNeurons; i++)
                //    layer.Gamma.D[i][0] = 1.0;
                //for (int i = 0; i < layer.numNeurons; i++)
                //    layer.Beta.D[i][0] = 0.0;
                layer.Mu.Clear();
                layer.Var.Clear();
                layer.dGamma.Clear();
                layer.dBeta.Clear();
                for (int b = 0; b < layer.batchSize; b++)
                {
                    layer.Xhat[b].Clear();
                    layer.GradW[b].Clear();
                    layer.GradB[b].Clear();
                }
            }
        }
        void UpdateKernelsWeightsBiases(double learningRate, int batchSize)
        {
            //---------------update kernels and weights-----
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
                layer.W = layer.W - GradW.Mul(1.0 / batchSize).Mul(learningRate);
                layer.B = layer.B - GradB.Mul(1.0 / batchSize).Mul(learningRate);
                layer.Gamma = layer.Gamma - layer.dGamma.Mul(1.0 / batchSize).Mul(learningRate);
                layer.Beta = layer.Beta - layer.dBeta.Mul(1.0 / batchSize).Mul(learningRate);
            }
        }
    }
    
}
