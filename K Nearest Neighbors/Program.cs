using Emgu.CV;
using Emgu.CV.ML;
using Emgu.CV.Structure;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

/// <summary>
/// 一个样本与数据集中的 k 个样本最相似，
/// 如果这 k 个样本中的大多数属于某一个类别，则该样本也属于这个类别，
/// 即每个样本都可以用它最接近的 k 个邻居来代表。
/// </summary>
namespace K_Nearest_Neighbors
{
    internal class Program
    {
        static void Main(string[] args)
        {
            BaseKNN();
        }

        private static void BaseKNN()
        {
            int K = 10;//K的值会在3～10之间。
            int trainSampleCount = 200;//样本数量

            #region Generate the training data and classes

            Matrix<float> trainData = new Matrix<float>(trainSampleCount, 2);
            Matrix<float> trainClasses = new Matrix<float>(trainSampleCount, 1);

            Image<Bgr, Byte> img = new Image<Bgr, byte>(500, 500);

            Matrix<float> sample = new Matrix<float>(1, 2);

            Matrix<float> trainData1 = trainData.GetRows(0, trainSampleCount >> 1, 1);
            trainData1.SetRandNormal(new MCvScalar(300), new MCvScalar(50));
            Matrix<float> trainData2 = trainData.GetRows(trainSampleCount >> 1, trainSampleCount, 1);
            trainData2.SetRandNormal(new MCvScalar(500), new MCvScalar(50));

            Matrix<float> trainClasses1 = trainClasses.GetRows(0, trainSampleCount >> 1, 1);
            trainClasses1.SetValue('A');
            Matrix<float> trainClasses2 = trainClasses.GetRows(trainSampleCount >> 1, trainSampleCount, 1);
            trainClasses2.SetValue('B');
            #endregion

            Matrix<float> results, neighborResponses;
            results = new Matrix<float>(sample.Rows, 1);
            neighborResponses = new Matrix<float>(sample.Rows, K);
            //dist = new Matrix<float>(sample.Rows, K);

            using (KNearest knn = new KNearest())
            {
                knn.DefaultK = K;
                knn.IsClassifier = true;
                knn.Train(trainData, Emgu.CV.ML.MlEnum.DataLayoutType.RowSample, trainClasses);

                for (int i = 0; i < img.Height; i++)
                {
                    for (int j = 0; j < img.Width; j++)
                    {
                        sample.Data[0, 0] = j;
                        sample.Data[0, 1] = i;

                        // estimates the response and get the neighbors' labels
                        float response = knn.Predict(sample); //knn.FindNearest(sample, K, results, null, neighborResponses, null);

                        int accuracy = 0;
                        // compute the number of neighbors representing the majority
                        for (int k = 0; k < K; k++)
                        {
                            if (neighborResponses.Data[0, k] == response)
                                accuracy++;
                        }
                        //Console.WriteLine($"Row = {i}, Col = {j},response = {response},accuracy = {accuracy}");
                        // highlight the pixel depending on the accuracy (or confidence)
                        img[i, j] =
                           response == 1 ?
                              (accuracy > 5 ? new Bgr(90, 0, 0) : new Bgr(90, 40, 0)) :
                              (accuracy > 5 ? new Bgr(0, 90, 0) : new Bgr(40, 90, 0));
                    }
                }

            }

            // display the original training samples
            for (int i = 0; i < (trainSampleCount >> 1); i++)
            {
                PointF p1 = new PointF(trainData1[i, 0], trainData1[i, 1]);
                img.Draw(new CircleF(p1, 2.0f), new Bgr(255, 100, 100), -1);
                PointF p2 = new PointF(trainData2[i, 0], trainData2[i, 1]);
                img.Draw(new CircleF(p2, 2.0f), new Bgr(100, 255, 100), -1);
            }

            Emgu.CV.UI.ImageViewer.Show(img);
        }

        private static void KNN_3Clusters()
        {
            int K = 10;//K的值会在3～10之间。
            int trainSampleCount = 100;//样本数量

            #region Generate the training data and classes

            Matrix<float> trainData = new Matrix<float>(trainSampleCount, 2);
            Matrix<float> trainClasses = new Matrix<float>(trainSampleCount, 1);

            Image<Bgr, Byte> img = new Image<Bgr, byte>(500, 500);

            Matrix<float> sample = new Matrix<float>(1, 2);

            Matrix<float> trainData1 = trainData.GetRows(0, trainSampleCount / 3, 1);
            trainData1.SetRandNormal(new MCvScalar(300), new MCvScalar(50));
            Matrix<float> trainData2 = trainData.GetRows(trainSampleCount / 3, 2 * trainSampleCount / 3, 1);
            trainData2.SetRandNormal(new MCvScalar(400), new MCvScalar(50));
            Matrix<float> trainData3 = trainData.GetRows(2 * trainSampleCount / 3, trainSampleCount, 1);
            trainData3.SetRandNormal(new MCvScalar(500), new MCvScalar(50));

            Matrix<float> trainClasses1 = trainClasses.GetRows(0, trainSampleCount / 3, 1);
            trainClasses1.SetValue(1);
            Matrix<float> trainClasses2 = trainClasses.GetRows(trainSampleCount / 3, 2 * trainSampleCount / 3, 1);
            trainClasses2.SetValue(2);
            Matrix<float> trainClasses3 = trainClasses.GetRows(2 * trainSampleCount / 3, trainSampleCount, 1);
            trainClasses3.SetValue(3);
            #endregion

            Matrix<float> results, neighborResponses;
            results = new Matrix<float>(sample.Rows, 1);
            neighborResponses = new Matrix<float>(sample.Rows, K);
            //dist = new Matrix<float>(sample.Rows, K);

            using (KNearest knn = new KNearest())
            {
                knn.DefaultK = K;
                knn.IsClassifier = true;
                knn.Train(trainData, Emgu.CV.ML.MlEnum.DataLayoutType.RowSample, trainClasses);

                for (int i = 0; i < img.Height; i++)
                {
                    for (int j = 0; j < img.Width; j++)
                    {
                        sample.Data[0, 0] = j;
                        sample.Data[0, 1] = i;

                        // estimates the response and get the neighbors' labels
                        float response = knn.Predict(sample); //knn.FindNearest(sample, K, results, null, neighborResponses, null);
                        Console.WriteLine($"Row = {i}, Col = {j},response: = {response}");                      int accuracy = 0;
                        // compute the number of neighbors representing the majority
                        for (int k = 0; k < K; k++)
                        {
                            if (neighborResponses.Data[0, k] == response)
                                accuracy++;
                        }
                        // highlight the pixel depending on the accuracy (or confidence)
                        img[i, j] =
                           response == 1 ?
                              (accuracy > 5 ? new Bgr(90, 0, 0) : new Bgr(90, 40, 0)) :
                              (accuracy > 5 ? new Bgr(0, 90, 0) : new Bgr(40, 90, 0));
                    }
                }

            }

            // display the original training samples
            for (int i = 0; i < (trainSampleCount /3); i++)
            {
                PointF p1 = new PointF(trainData1[i, 0], trainData1[i, 1]);
                img.Draw(new CircleF(p1, 2.0f), new Bgr(255, 100, 100), -1);
                PointF p2 = new PointF(trainData2[i, 0], trainData2[i, 1]);
                img.Draw(new CircleF(p2, 2.0f), new Bgr(100, 255, 100), -1);
                PointF p3 = new PointF(trainData3[i, 0], trainData3[i, 1]);
                img.Draw(new CircleF(p3, 2.0f), new Bgr(0, 255, 0), -1);
            }

            Emgu.CV.UI.ImageViewer.Show(img);
        }
    }
}
