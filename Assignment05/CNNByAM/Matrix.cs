using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading.Tasks;

namespace MatrixLibByAM
{
    [Serializable]
    class Matrix
    {
        // uses jagged array as it is faster
        public double [][] D = null;  // data in matrix
        public int Rows = 0;
        public int Cols = 0;
        public Matrix (int rows, int cols)
        {
            this.Rows = rows;
            this.Cols = cols;
            // create a jagged array as it is faster
            D = new double[rows][];
            for (int i = 0; i < rows; i++)
                D[i] = new double[cols];
        }

        public Matrix Transpose()
        {
            Matrix Res = new Matrix(Cols, Rows);
            for (int i = 0; i < Rows;i++)
                for (int j = 0; j < Cols; j++)
                    Res.D[j][i] = D[i][j];
            return Res;
        }

        public static Matrix operator +(Matrix lhs, Matrix rhs)
        {
            Matrix Res = new Matrix(lhs.Rows, lhs.Cols);
            for (int i = 0; i < lhs.Rows; i++)
                for (int j = 0; j < lhs.Cols; j++)
                    Res.D[i][j] = lhs.D[i][j] + rhs.D[i][j];
            return Res;
        }

        public static Matrix operator *(Matrix lhs, Matrix rhs)
        {
            Matrix Res = new Matrix(lhs.Rows, rhs.Cols);
            for (int i = 0; i < lhs.Rows; i++)
           // Parallel.For(0, lhs.Rows, (i) =>
            {
                for (int j = 0; j < rhs.Cols; j++)
                    for (int k = 0; k < lhs.Cols; k++)
                        Res.D[i][j] = Res.D[i][j] + lhs.D[i][k] * rhs.D[k][j];
            }
            return Res;
        }

        public static Matrix operator -(Matrix lhs, Matrix rhs)
        {
            Matrix Res = new Matrix(lhs.Rows, lhs.Cols);
            for (int i = 0; i < lhs.Rows; i++)
                for (int j = 0; j < lhs.Cols; j++)
                    Res.D[i][j] = lhs.D[i][j] - rhs.D[i][j];
            return Res;
        }

        public void AddInPlace(double val)  // add a number to each element
        {
            for (int i = 0; i < this.Rows; i++)
                for (int j = 0; j < this.Cols; j++)
                    D[i][j] = D[i][j] + val;
        }
        public Matrix Add(double val)  // add a number to each element
        {
            Matrix Res = this.Clone();
            for (int i = 0; i < this.Rows; i++)
                for (int j = 0; j < this.Cols; j++)
                    Res.D[i][j] = Res.D[i][j] + val;
            return Res;
        }

        public void MulInPlace(double val)  // multiplies a number to each element
        {
            for (int i = 0; i < this.Rows; i++)
                for (int j = 0; j < this.Cols; j++)
                    D[i][j] = D[i][j] * val;
        }

        public Matrix Mul(double val)  // multiplies a number to each element
        {
            Matrix Res = this.Clone();
            for (int i = 0; i < this.Rows; i++)
                for (int j = 0; j < this.Cols; j++)
                    Res.D[i][j] = Res.D[i][j] * val;
            return Res;
        }

        public Matrix ElementByElementMul(Matrix A)  // multiplies element by element
        {
            Matrix Res = new Matrix(A.Rows, A.Cols);
            for (int i = 0; i < A.Rows; i++)
                for (int j = 0; j < A.Cols; j++)
                    Res.D[i][j] = D[i][j] * A.D[i][j];
            return Res;
        }

        public Matrix SoftMax()  // apply exp to each element
        {
            Matrix Res = new Matrix(Rows, Cols);
            double sum = 0;
            for (int i = 0; i < Rows; i++)
            {
                Res.D[i][0] = Math.Exp(D[i][0]);
                sum += Res.D[i][0];
            }
            //Console.WriteLine("sum = " + sum.ToString());
            for (int i = 0; i < Rows; i++)
            {
                if (sum > 0.0001)
                    Res.D[i][0] = Res.D[i][0] / sum;
                else
                    Res.D[i][0] = 0.1; // some limit
            }
            return Res;
        }

        public Matrix Sigmoid()  // apply sigmoid to each element
        {
            Matrix Res = new Matrix(Rows, Cols);
            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Cols; j++)
                    Res.D[i][j] = 1.0 / (1 + Math.Exp(-D[i][j]));
            }
            return Res;
        }
        public Matrix RELU()  // apply RELU to each element
        {
            Matrix Res = new Matrix(Rows, Cols);
            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Cols; j++)
                {
                    if (D[i][j] <= 0)
                        Res.D[i][j] = 0;
                    else
                        Res.D[i][j] = D[i][j];
                }
            }
            return Res;
        }

        public Matrix AvgPool()  
        {
            Matrix Res = new Matrix(Rows / 2, Cols / 2);
            for (int i = 0; i < Rows / 2; i++)
            {
                for (int j = 0; j < Cols / 2; j++)
                    Res.D[i][j] = (D[i * 2][j * 2] + D[i * 2][j * 2 + 1] + D[i * 2 + 1][j * 2] + D[i * 2 + 1][j * 2 + 1]) / 4.0;
            }
            return Res;
        }

        public Matrix Correlation(Matrix kernel)
        {   // no padding, assumes kernel is a square matrix, no flip of kernel
            int k = kernel.Cols;
            int m = Rows;
            int n = Cols;
            Matrix Res = new Matrix(m - (k - 1), n - (k - 1)); // output size is more 
            for (int i = 0; i < m - (k - 1); i++)
            {
                for (int j = 0; j < n - (k - 1); j++)
                {
                    double sum = 0;
                    for (int ki = 0; ki < k; ki++) // iterate over kernel
                    {
                        for (int kj = 0; kj < k; kj++)
                        {
                            // ***todo*** see if this check is unnecessary
                            if (((i + ki) >= 0) && ((i + ki) < Rows) && ((j + kj) >= 0) && ((j + kj) < Cols))
                            {
                                double data = D[i + ki][j + kj];
                                double kval = kernel.D[ki][kj];
                                sum += data * kval;
                            }
                        }
                    }
                    Res.D[i][j] = sum;
                }
            }
            return Res;
        }


        public Matrix Convolution(Matrix kernel)
        {
            Matrix rotKernel = kernel.RotateBy90().RotateBy90();
            return this.Correlation(rotKernel);
        }

        public Matrix ConvolutionFull(Matrix kernel)
        {
            Matrix rotKernel = kernel.RotateBy90().RotateBy90();
            return this.CorrelationFull(rotKernel);
        }

        public Matrix CorrelationFull(Matrix kernel)
        {   // assumes kernel is a square matrix, no flip of kernel
            int k = kernel.Cols;
            int m = Rows;
            int n = Cols;
            Matrix Res = new Matrix(m + (k - 1), n + (k - 1)); // output size is more 
            for (int i = 0; i < m + (k - 1); i++)
            {
                for (int j = 0; j < n + (k - 1); j++)
                {
                    double sum = 0;
                    for (int ki = -(k - 1); ki <= 0; ki++) // iterate over kernel
                    {
                        for (int kj = -(k - 1); kj <= 0; kj++)
                        {
                            // ***todo*** see if this check is unnecessary
                            if (((i + ki) >= 0) && ((i + ki) < Rows) && ((j + kj) >= 0) && ((j + kj) < Cols))
                            {
                                double data = D[i + ki][j + kj];
                                double kval = kernel.D[ki + (k - 1)][kj + (k - 1)];
                                sum += data * kval;
                            }
                        }
                    }
                    Res.D[i][j] = sum;
                }
            }
            return Res;
        }

        public Matrix RotateBy90()
        {
            Matrix Res = new Matrix(Rows, Cols);
            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Cols; j++)
                {
                    Res.D[i][j] = D[Rows - j - 1][i];
                }
            }
            return Res;
        }

        public Matrix Flatten()
        {  // converts mxn matrix to ((mxn) x 1) vecttor
            Matrix Res = new Matrix(Rows * Cols, 1);
            int count = 0;
            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Cols; j++)
                {
                    Res.D[count][0] = D[i][j];
                    count++;
                }
            }
            return Res;
        }

        public void Clear()  // zero out
        {
            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Cols; j++)
                {
                    D[i][j] = 0;
                }
            }
        }

        public Matrix Clone()
        {
            return GenericCopier<Matrix>.DeepCopy(this);
        }
    }

    public static class GenericCopier<T>    //deep copy a list
    {
        public static T DeepCopy(object objectToCopy)
        {
            using (MemoryStream memoryStream = new MemoryStream())
            {
                BinaryFormatter binaryFormatter = new BinaryFormatter();
                binaryFormatter.Serialize(memoryStream, objectToCopy);
                memoryStream.Seek(0, SeekOrigin.Begin);
                return (T)binaryFormatter.Deserialize(memoryStream);
            }
        }
    }
}
