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
    class FeatureMap
    {  // featuremap for layer2 in cnn and onwards
        public FeatureMap(int inputDataSize, PoolingType poolingType, ActivationType activationType, int batchSize)
        {
            this.inputDataSize = inputDataSize;
            this.poolingType = poolingType;
            this.activationType = activationType;
            this.batchSize = batchSize;
            DeltaSS = new Matrix[batchSize];
            DeltaCV = new Matrix[batchSize];
            OutPutSS = new Matrix[batchSize];
            ActCV = new Matrix[batchSize];
            APrime = new Matrix[batchSize];
            Sum = new Matrix[batchSize];
        }
        Random rand = new Random((int)(DateTime.Now.Ticks));  // 5000 = seed for repeatable results
        PoolingType poolingType;
        public ActivationType activationType;
        int inputDataSize;
        int batchSize = 0;
        public Matrix[] DeltaSS { get; set; }   // subsampling deltas
        public Matrix[] DeltaCV { get; set; }   // layer deltas
        public Matrix[] OutPutSS { get; set; }  // subsampling layer output
        public Matrix[] ActCV { get; set; }  // Activation function outputs 
        public Matrix[] APrime { get; set; }  // Aprime 
        public Matrix[] Sum { get; set; }  // result after convol, then +  bias
        public double Bias { get; set; } // one bias for the feature map
        public double BiasGrad { get; set; } // one bias grad for the feature map

        public Matrix Evaluate(Matrix inputData, int batchIndex)
        {
            int c2Size = inputData.Rows;
            Matrix Res = new Matrix(c2Size, c2Size);
            Sum[batchIndex] = inputData.Add(Bias);
            if (activationType == ActivationType.SIGMOID)
            {
                ActCV[batchIndex] = Sum[batchIndex].Sigmoid();
                APrime[batchIndex] = 
                    ActCV[batchIndex].ElementByElementMul(ActCV[batchIndex].Mul(-1).Add(1));
            }
            if (activationType == ActivationType.RELU)
            {
                ActCV[batchIndex] = Sum[batchIndex].RELU(); // no aprime for relu, delta is made zero for neg sums
            }
            if (poolingType == PoolingType.AVGPOOLING)
                Res = ActCV[batchIndex].AvgPool();
            OutPutSS[batchIndex] = Res;
            return Res;
        }

    }
}

