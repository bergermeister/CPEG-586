using MatrixLibByAM;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNNAM
{
    [Serializable]
    class Layer
    {
        // represents one layer in a regular neural network
        public Layer(int numNeurons, int inputSize, ActivationType activationType, int batchSize, 
            double dropout = 1.0, double momentumBN = 0.8)
        {  // input size = number of inputs feeding to this layer
            this.numNeurons = numNeurons;
            this.batchSize = batchSize;
            this.Dropout = dropout;
            W = new Matrix(numNeurons, inputSize);
            B = new Matrix(numNeurons, 1);  // biases
            Ivar = new Matrix(numNeurons, 1);  // inverse of variance
            Mu = new Matrix(numNeurons, 1); // batch mean
            Beta = new Matrix(numNeurons, 1);  // initialize to zeros
            Gamma = new Matrix(numNeurons, 1); // initialize to 1's
            for (int i = 0; i < numNeurons; i++)
                Gamma.D[i][0] = 1.0;
            Xhat = new Matrix[batchSize];
            for (int i = 0; i < Xhat.Length; i++)
                Xhat[i] = new Matrix(numNeurons, 1);
            dBeta = new Matrix(numNeurons, 1);
            dGamma = new Matrix(numNeurons, 1);
            Var = new Matrix(numNeurons, 1);// batch variance
            RunningMu = new Matrix(numNeurons, 1); // running batch mean
            RunningVar = new Matrix(numNeurons, 1);// running batch variance
            this.MomentumBN = momentumBN;

            InitializeWeights(W);
            Delta = new Matrix[batchSize];
            for (int i = 0; i < Delta.Length; i++)
                Delta[i] = new Matrix(numNeurons, 1); // for storing deltas
            GradW = new Matrix[batchSize];
            for (int i = 0; i < GradW.Length; i++)
                GradW[i] = new Matrix(numNeurons, inputSize);
            GradB = new Matrix[batchSize];
            for (int i = 0; i < GradB.Length; i++)
                GradB[i] = new Matrix(numNeurons, 1);
            Sum = new Matrix[batchSize];
            for (int i = 0; i < Sum.Length; i++)
                Sum[i] = new Matrix(numNeurons, 1);
            DropM = new Matrix[batchSize];
            for (int i = 0; i < Sum.Length; i++)
                DropM[i] = new Matrix(numNeurons, 1);
            A = new Matrix[batchSize];
            for (int i = 0; i < A.Length; i++)
                A[i] = new Matrix(numNeurons, 1);
            APrime = new Matrix[batchSize];
            for (int i = 0; i < APrime.Length; i++)
                APrime[i] = new Matrix(numNeurons, 1);
            this.activationType = activationType;
        }
        public ActivationType activationType = ActivationType.SIGMOID;
        public double Dropout = 1.0;
        public Random rand = new Random();  // 5000 = seed for repeatable results
        public int numNeurons;
        public int batchSize = 0;
        public Matrix W = null;
        public Matrix B = null;
        public Matrix[] A = null;
        public Matrix[] APrime = null;
        public Matrix[] Sum = null;
        public Matrix[] DropM = null;  // dropout matrix --contains zeros or 1/p
        public Matrix[] GradW = null;
        public Matrix[] GradB = null;
        public Matrix[] Delta = null;

        public Matrix Mu = null;   // for implementing bach norm
        public Matrix Var = null;   // for implementing bach norm
        public Matrix[] Xhat = null;
        public Matrix Ivar = null;
        public Matrix RunningMu = null;
        public Matrix RunningVar = null;
        public Matrix Gamma = null;
        public Matrix Beta = null;
        public Matrix dGamma = null;
        public Matrix dBeta = null;
        public double MomentumBN = 0;
        public double epsilon = 1e-8;

        public void InitializeDropoutMatrix(Matrix Dm)
        {
            if (Dropout < 1.0)
            {
                for (int i = 0; i < Dm.Rows; i++)
                {
                    for (int j = 0; j < Dm.Cols; j++)
                    {
                        double num = rand.NextDouble();
                        if (num < Dropout)
                            Dm.D[i][j] = 1/Dropout;
                        else
                            Dm.D[i][j] = 0;
                    }
                }
            }
        }

        void InitializeWeights(Matrix W)
        {
            for(int i = 0; i < W.Rows; i++)
            {
                for (int j = 0; j < W.Cols; j++)
                {
                    double num = rand.NextDouble();
                    if (num < 0.5)
                        W.D[i][j] = rand.NextDouble() * 0.1;
                    else
                        W.D[i][j] = rand.NextDouble() * -0.1;

                }
            }
            // bias initialized to zero by default
        }
        public Matrix Evaluate(Matrix inputData, int bi, bool useBatchNorm, bool doBNTestMode = false)  // bi = batchIndex
        {
            Sum[bi] = W * inputData + B;
            /*--------batch norm computations----------
                x_minus_mu = x - batch_mean
                ivar = 1.0 / np.sqrt(batch_var+eps)
                x_hat = x_minus_mu * ivar
                out = gamma * x_hat + beta*/
            if ((useBatchNorm == true) && (doBNTestMode == false))
            {
                Matrix x_minus_mu = Sum[bi] - Mu;
                Xhat[bi] = x_minus_mu.ElementByElementMul(Ivar);
                Sum[bi] = Xhat[bi].ElementByElementMul(Gamma) + Beta;
            }
            if ((useBatchNorm == true) && (doBNTestMode == true))
            {
                Matrix x_minus_Runningmu = Sum[bi] - RunningMu;
                for (int kk = 0; kk < RunningVar.Rows; kk++)
                    Xhat[bi].D[kk][0] = x_minus_Runningmu.D[kk][0] / Math.Sqrt(RunningVar.D[kk][0] + epsilon);
                Sum[bi] = Xhat[bi].ElementByElementMul(Gamma) + Beta;
            }
            //-------apply dropout----------------------
            if (Dropout < 1.0)
            {
                InitializeDropoutMatrix(DropM[bi]);
                Sum[bi] = Sum[bi].ElementByElementMul(DropM[bi]);
            }
            //--------------------------------------------
            if (activationType == ActivationType.SIGMOID)
            {
                A[bi] = Sum[bi].Sigmoid();
                this.APrime[bi] = A[bi].ElementByElementMul((A[bi].Mul(-1).Add(1)));  // A * (1 - A)
            }

            if (activationType == ActivationType.RELU)
                A[bi] = Sum[bi].RELU();  // no aprime for relu - delta is computed accordingly during training
            if (activationType == ActivationType.SOFTMAX)
                A[bi] = Sum[bi].SoftMax();

            return A[bi];
        }
    }
}
