using CNNAM;
using MatrixLibByAM;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace CNNByAM
{
    public partial class Form1 : Form
    {
        DeepCNN dcnn = null;
        Network nn = null;
        public Form1()
        {
            InitializeComponent();
        }

        private void btnTrainCNN_Click(object sender, EventArgs e)
        {
            int batchSize = 5;
            int numFeatureMapsLayer1 = 6;  // returns accuracy of around 92% on 1000 training data
                                           // and 10000 test data, 6, 12, l1=50, l2 = 10 architecture
            int numFeatureMapsLayer2 = 12; // 12 feature maps in second layer
            CNNLayer C1 = new CNNLayer(numFeatureMapsLayer1, 1, 28, 5, // 28 = input size
                PoolingType.AVGPOOLING, ActivationType.RELU,batchSize); // 1 = featuremaps in prev layer
                                                              // since C1 connects to input, the number f featuremaps is treated as 1
                                                              // 28 is the input size, 5 is the convol. kernel size
            CNNLayer C2 = new CNNLayer(numFeatureMapsLayer2, numFeatureMapsLayer1, 12, 5,
                PoolingType.AVGPOOLING, ActivationType.RELU,batchSize);
            List<CNNLayer> CNNList = new List<CNNLayer> { C1, C2 };

            // for MNIST, the second CNN layer produces an output of 4x4
            Layer l1 = new Layer(50, 4 * 4 * numFeatureMapsLayer2, ActivationType.RELU,batchSize,0.8);
            // 50 is the number of hidden layer neurons
            // 4x4xnumFeatureMapsLayer2 is the flatten size to the first layer of regular NN
            Layer l2 = new Layer(10, 50, ActivationType.SOFTMAX,batchSize);
            List<Layer> NNLayerList = new List<Layer> { l1, l2 };

            var data = ReadMNISTTrainingData();

            dcnn = new DeepCNN(CNNList, NNLayerList, data.Item1, data.Item2,batchSize);
            Stopwatch sw = new Stopwatch();
            sw.Start();
            dcnn.Train(30, 0.1, batchSize);  // 25 epochs, 0.1 learning rate, 10 batch size
            sw.Stop();
            double accuracyPercent = ComputeAccuracy(true);
            MessageBox.Show("done training.., accuracy = " + (accuracyPercent * 100).ToString() +
            "\nTraining Time = " + (sw.ElapsedMilliseconds / 1000) + " seconds");

           
        }

        double ComputeAccuracy(bool deepCNN)
        {
            String testDir = "D:\\csharp2016\\DeepLearning\\MNIST\\data\\TestAll10000\\";
            //string testDir = @"D:\csharp2018\DataSets\MNIST\MNISTBMP\data\TestAll10000";
            DirectoryInfo dinfo = new DirectoryInfo(testDir);
            int accuracyCount = 0;
            foreach (FileInfo fi in dinfo.GetFiles())
            {
                Matrix Img = ReadOneImage(fi.FullName);
                FileInfo finfo = new FileInfo(fi.FullName);
                Char output = finfo.Name[0];
                int classLabel = (Convert.ToInt16(output) - 48); //will only work with numbers 0-9
                Matrix res = null;
                if (deepCNN)
                    res = dcnn.Evaluate(Img,0);
                else
                    res = nn.Evaluate(Img.Flatten(),0,true,true);
                double max = -1;
                int index = -1;
                for (int i = 0; i < res.Rows; i++)
                {
                    if (res.D[i][0] > max)
                    {
                        max = res.D[i][0];
                        index = i;
                    }
                }
                if (index == classLabel)
                    accuracyCount++;
            }
            return accuracyCount / 10000.0;
        }
        Tuple<List<Matrix>, List<Matrix>> ReadMNISTTrainingData()
        {
            String trainDir = "D:\\DeepLearning\\mnistdata\\data\\Training1000\\";
            //string trainDir = @"D:\csharp2018\DataSets\MNIST\MNISTBMP\Training1000";
            //string trainDir = @"D:\csharp2016\DeepLearning\MNIST\data\TrainingAll60000";
            //string trainDir = @"D:\csharp2018\DataSets\MNIST\MNISTBMP\data\TrainingAll60000";
            int numFiles = 0;
            int dataIndex = 0;
            DirectoryInfo dinfo = new DirectoryInfo(trainDir);
            List<Matrix> InputDataList = new List<Matrix>();
            List<Matrix> OutputLabelsList = new List<Matrix>();
            //count the number of files 
            foreach (FileInfo fi in dinfo.GetFiles())
                numFiles++;

            foreach (FileInfo fi in dinfo.GetFiles())
            {
                Matrix Img = new Matrix(28, 28);
                String fname = fi.FullName;
                Bitmap bmp = new Bitmap(Image.FromFile(fname));
                if (ImageProc.IsGrayScale(bmp) == false) //make sure it is grayscale 
                {
                    ImageProc.ConvertToGray(bmp);
                }
                for (int i = 0; i < bmp.Width; i++)
                {
                    for (int j = 0; j < bmp.Height; j++)
                    {   // rescale between 0 and 1
                        Img.D[i][j] = bmp.GetPixel(i, j).R / 255.0;
                    }
                }
                InputDataList.Add(Img);
                Matrix outputLabel = new Matrix(10, 1);
                String s1 = fi.Name;
                Char output = s1[0];
                int classLabel = (Convert.ToInt16(output) - 48); //will only work with numbers 0-9
                outputLabel.D[classLabel][0] = 1;  // others are 0 by default
                OutputLabelsList.Add(outputLabel);

                dataIndex++;
                if ((dataIndex % 500) == 0)
                    Console.WriteLine("iter: " + dataIndex);
            }
            return new Tuple<List<Matrix>, List<Matrix>>(InputDataList, OutputLabelsList);
        }

        Matrix ReadOneImage(string fname)
        {
            Matrix Img = new Matrix(28, 28);
            Bitmap bmp = new Bitmap(Image.FromFile(fname));
            if (ImageProc.IsGrayScale(bmp) == false) //make sure it is grayscale 
            {
                ImageProc.ConvertToGray(bmp);
            }
            for (int i = 0; i < bmp.Width; i++)
            {
                for (int j = 0; j < bmp.Height; j++)
                {   // rescale between 0 and 1
                    Img.D[i][j] = bmp.GetPixel(i, j).R / 255.0;
                }
            }
            return Img;
        }

        private void btnTestDigitAfterTraining_Click(object sender, EventArgs e)
        {
            try
            {
                OpenFileDialog ofd = new OpenFileDialog();
                ofd.InitialDirectory = "D:\\csharp2016\\DeepLearning\\MNIST\\data\test";
                if (ofd.ShowDialog() == DialogResult.OK)
                {
                    picTest.Image = new Bitmap(ofd.FileName);
                    Matrix Img = ReadOneImage(ofd.FileName);
                    FileInfo fi = new FileInfo(ofd.FileName);
                    Char output = fi.Name[0];
                    int classLabel = (Convert.ToInt16(output) - 48); //numbers 0-9
                    Matrix res = null;
                    if (dcnn != null)
                        res = dcnn.Evaluate(Img, 0);
                    else
                        res = nn.Evaluate(Img.Flatten(), 0, true, true);
                   // Matrix res = dcnn.Evaluate(Img);
                    if (res.D[classLabel][0] > 0.5)
                    {
                        MessageBox.Show("matched ..Label = " + classLabel.ToString());
                    }
                    else
                    {
                        MessageBox.Show("incorrect match..");
                    }
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }

        private void btnSaveNetworkAfterTraining_Click(object sender, EventArgs e)
        {
            SaveFileDialog ofd = new SaveFileDialog();
            ofd.InitialDirectory = "c:\\temp";
            if (ofd.ShowDialog() == DialogResult.OK)
            {
                string fname = ofd.FileName;
                FileInfo fi = new FileInfo(fname);
                Stream str = fi.Open(FileMode.OpenOrCreate, FileAccess.Write);
                BinaryFormatter bf = new BinaryFormatter();
              //  bf.Serialize(str, dcnn);
                str.Close();
                MessageBox.Show("deep CNN saved successfully..");
            }
        }

        private void btnLoadTrainedNetwork_Click(object sender, EventArgs e)
        {
            OpenFileDialog ofd = new OpenFileDialog();
            ofd.InitialDirectory = "c:\\temp";
            if (ofd.ShowDialog() == DialogResult.OK)
            {
                string fname = ofd.FileName;
                FileInfo fi = new FileInfo(fname);
                Stream str = fi.Open(FileMode.Open, FileAccess.Read);
                BinaryFormatter bf = new BinaryFormatter();
               // dcnn = (DeepCNN)bf.Deserialize(str);
                str.Close();
                MessageBox.Show("deep CNN restored successfully..");
            }
        }

        private void btnTest_Click(object sender, EventArgs e)
        {
            MatrixLibByAM.Matrix M = new MatrixLibByAM.Matrix(4, 6);
            M.D[2][3] = 7.5;
            M.D[3][5] = 9.2;
            MessageBox.Show(M.D[3][5].ToString());
        }

        private void btnRegularNN_Click(object sender, EventArgs e)
        {
            Layer l1 = new Layer(50, 28*28, ActivationType.SIGMOID,10);
           // 50 is the number of hidden layer neurons
           // 4x4xnumFeatureMapsLayer2 is the flatten size to the first layer of regular NN
           Layer l2 = new Layer(10, 50, ActivationType.SOFTMAX,10);
           List<Layer> NNLayerList = new List<Layer> { l1, l2 };

           var data = ReadMNISTTrainingData();

           nn = new Network(NNLayerList, data.Item1, data.Item2);
           Stopwatch sw = new Stopwatch();
           sw.Start();
           nn.Train(50, 0.1, 10,true);  // 30 epochs, 0.1 learning rate, 5 batch size
           sw.Stop();
           long millisecs = sw.ElapsedMilliseconds;
           double accuracyPercent = ComputeAccuracy(false);
           MessageBox.Show("done training.., accuracy = " + (accuracyPercent * 100).ToString() +
               "\nTraining Time = " + (millisecs/1000) + " seconds"); 
        }
    }
}
