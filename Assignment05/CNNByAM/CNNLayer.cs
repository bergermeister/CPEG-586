using CNNAM;
using MatrixLibByAM;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNNByAM
{
    [Serializable]
    class CNNLayer
    {
        public CNNLayer(int numFeatureMaps, int numPrevLayerFeatureMaps, int inputSize,
            int kernelSize, PoolingType poolingType, ActivationType activationType, int batchSize)
        {
            this.batchSize = batchSize;
            this.ConvolSums = new Matrix[batchSize,numFeatureMaps];
            this.kernelSize = kernelSize;
            this.numFeatureMaps = numFeatureMaps;
            this.numPrevLayerFeatureMaps = numPrevLayerFeatureMaps;
            this.ConvolResults = new Matrix[batchSize,numPrevLayerFeatureMaps, numFeatureMaps];
            int convOutputSize = inputSize - kernelSize + 1;
            for (int i = 0; i < batchSize; i++)
                for (int j = 0; j < numFeatureMaps; j++)
                    ConvolSums[i,j] = new Matrix(convOutputSize, convOutputSize);
            Kernels = new Matrix[numPrevLayerFeatureMaps, numFeatureMaps];
            KernelGrads = new Matrix[numPrevLayerFeatureMaps, numFeatureMaps];
            InitMatrix2DArray(ref Kernels,numPrevLayerFeatureMaps, numFeatureMaps,kernelSize);
            InitMatrix2DArray(ref KernelGrads, numPrevLayerFeatureMaps, numFeatureMaps, kernelSize);

            InitializeKernels();
            for (int i = 0; i < numFeatureMaps; i++)
            {
                FeatureMap fmp = new FeatureMap(convOutputSize, poolingType, activationType,batchSize);
                FeatureMapList.Add(fmp);
            }
        }

        int kernelSize;
        // will do prev maps * numFeatureMap convolutions
        // for the first CNN layer, it will do 1xnumFeatureMaps convolutions
        // for subsequent layers, it will do prevLayerFeatureMapsxnumFeatureMaps
        // convolutions. Even though first CNN layer does not require a 2-D
        // array of matrices to store kernels, gradients, we will still use
        // 1xnumFeaturemaps arrays so that we can generalize to any number of 
        // CNN layers 
        int batchSize = 0;
        public Matrix[,] Kernels = null;
        public Matrix[,] KernelGrads = null;
        public Matrix[,,] ConvolResults = null;
        public Matrix[,] ConvolSums = null;
        int numFeatureMaps;
        int numPrevLayerFeatureMaps;
        public List<FeatureMap> FeatureMapList = new List<FeatureMap>();
        Random rand = new Random((int)(DateTime.Now.Ticks));  // 5000 = seed for repeatable results

        void InitializeKernels()  // initialize all kernels
        {
            for (int i = 0; i < Kernels.GetLength(0); i++)
                for (int j = 0; j < Kernels.GetLength(1); j++)
                    InitializeKernel(Kernels[i, j]);
        }

        public void Evaluate(List<Matrix>PrevLayerOutputList, int batchIndex)
        {   // inputs come from outputs of previous layer
            // do convolutions with outputs of feature maps from previous layer
            for (int p = 0; p < numPrevLayerFeatureMaps; p++)
            {
                for (int q = 0; q < numFeatureMaps; q++)
                //Parallel.For(0, numFeatureMaps, (q) =>
                {
                    ConvolResults[batchIndex,p, q] = PrevLayerOutputList[p].Convolution(Kernels[p, q]);
                }
            }
            // add convolution results
            for (int q = 0; q < FeatureMapList.Count; q++)
            //Parallel.For(0, FeatureMapList.Count, (q) => 
            {
                ConvolSums[batchIndex,q].Clear();
                for (int p = 0; p < PrevLayerOutputList.Count; p++)
                {
                    ConvolSums[batchIndex,q] = ConvolSums[batchIndex,q] + ConvolResults[batchIndex,p, q];
                }
            }
            // evaluate each feature map i.e., perform activation after adding bias
            for(int i = 0; i < FeatureMapList.Count;i++)
            {
                FeatureMapList[i].Evaluate(ConvolSums[batchIndex,i],batchIndex);
            }
        }

        void InitMatrix2DArray(ref Matrix[,] Mat, int dim1, int dim2, int matrixSize)
        {
            Mat = new Matrix[dim1,dim2];
            for (int i = 0; i < dim1; i++)
            {
                for (int j = 0; j < dim2; j++)
                {
                    Mat[i, j] = new Matrix(matrixSize, matrixSize);
                }
            }
        }

        void InitializeKernel(Matrix kernel)
        {
            for (int i = 0; i < kernel.Rows; i++)
            {
                for (int j = 0; j < kernel.Cols; j++)
                {
                    double num = rand.NextDouble();
                    if (num < 0.5)
                        kernel.D[i][j] = rand.NextDouble() * 0.1;  // D for data
                    else
                        kernel.D[i][j] = rand.NextDouble() * -0.1;
                }
            }
        }
    }
}
