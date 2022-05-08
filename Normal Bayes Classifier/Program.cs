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
/// A naive Bayes classifier is a term in Bayesian statistics dealing with a simple probabilistic classifier based on applying Bayes' theorem with strong (naive) independence assumptions. 
/// A more descriptive term for the underlying probability model would be "independent feature model".
///In simple terms, a naive Bayes classifier assumes that the presence (or absence) of a particular feature of a class is unrelated to the presence(or absence) of any other feature. 
///For example, a fruit may be considered to be an apple if it is red, round, and about 4" in diameter. 
///Even though these features depend on the existence of the other features, a naive Bayes classifier considers all of these properties to independently contribute to the probability that this fruit is an apple.
///Depending on the precise nature of the probability model, 
///naive Bayes classifiers can be trained very efficiently in a supervised learning setting. 
///In many practical applications, parameter estimation for naive Bayes models uses the method of maximum likelihood; in other words, one can work with the naive Bayes model without believing in Bayesian probability or using any Bayesian methods.
///In spite of their naive design and apparently over-simplified assumptions,
///naive Bayes classifiers often work much better in many complex real-world situations than one might expect. 
///Recently, careful analysis of the Bayesian classification problem has shown that there are some theoretical reasons for the apparently unreasonable efficacy of naive Bayes classifiers.
///[1] An advantage of the naive Bayes classifier is that it requires a small amount of training data to estimate the parameters (means and variances of the variables) necessary for classification.Because independent variables are assumed, only the variances of the variables for each class need to be determined and not the entire covariance matrix.
/// </summary>

namespace Normal_Bayes_Classifier
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Do5Clusters();
        }

        private static void Do3Clusters()
        {
            Bgr[] colors = new Bgr[] {
   new Bgr(0, 0, 255),
   new Bgr(0, 255, 0),
   new Bgr(255, 0, 0)};
            int trainSampleCount = 150;

            #region Generate the training data and classes
            Matrix<float> trainData = new Matrix<float>(trainSampleCount, 2);
            Matrix<int> trainClasses = new Matrix<int>(trainSampleCount, 1);

            Image<Bgr, Byte> img = new Image<Bgr, byte>(500, 500);

            Matrix<float> sample = new Matrix<float>(1, 2);

            Matrix<float> trainData1 = trainData.GetRows(0, trainSampleCount / 3, 1);
            trainData1.GetCols(0, 1).SetRandNormal(new MCvScalar(100), new MCvScalar(50));
            trainData1.GetCols(1, 2).SetRandNormal(new MCvScalar(300), new MCvScalar(50));

            Matrix<float> trainData2 = trainData.GetRows(trainSampleCount / 3, 2 * trainSampleCount / 3, 1);
            trainData2.SetRandNormal(new MCvScalar(400), new MCvScalar(50));

            Matrix<float> trainData3 = trainData.GetRows(2 * trainSampleCount / 3, trainSampleCount, 1);
            trainData3.GetCols(0, 1).SetRandNormal(new MCvScalar(300), new MCvScalar(50));
            trainData3.GetCols(1, 2).SetRandNormal(new MCvScalar(100), new MCvScalar(50));

            Matrix<int> trainClasses1 = trainClasses.GetRows(0, trainSampleCount / 3, 1);
            trainClasses1.SetValue(1);
            Matrix<int> trainClasses2 = trainClasses.GetRows(trainSampleCount / 3, 2 * trainSampleCount / 3, 1);
            trainClasses2.SetValue(2);
            Matrix<int> trainClasses3 = trainClasses.GetRows(2 * trainSampleCount / 3, trainSampleCount, 1);
            trainClasses3.SetValue(3);
            #endregion

            using (TrainData td = new TrainData(trainData, Emgu.CV.ML.MlEnum.DataLayoutType.RowSample, trainClasses))
            using (NormalBayesClassifier classifier = new NormalBayesClassifier())
            {
                //ParamDef[] defs = classifier.GetParams();
                classifier.Train(trainData, Emgu.CV.ML.MlEnum.DataLayoutType.RowSample, trainClasses);
                classifier.Clear();
                classifier.Train(td);
#if !NETFX_CORE
                String fileName = Path.Combine(Path.GetTempPath(), "normalBayes.xml");
                classifier.Save(fileName);
                if (File.Exists(fileName))
                    File.Delete(fileName);
#endif

                #region Classify every image pixel
                for (int i = 0; i < img.Height; i++)
                    for (int j = 0; j < img.Width; j++)
                    {
                        sample.Data[0, 0] = i;
                        sample.Data[0, 1] = j;
                        int response = (int)classifier.Predict(sample, null);

                        Bgr color = colors[response - 1];

                        img[j, i] = new Bgr(color.Blue * 0.5, color.Green * 0.5, color.Red * 0.5);
                    }
                #endregion
            }

            // display the original training samples
            for (int i = 0; i < (trainSampleCount / 3); i++)
            {
                PointF p1 = new PointF(trainData1[i, 0], trainData1[i, 1]);
                img.Draw(new CircleF(p1, 2.0f), colors[0], -1);
                PointF p2 = new PointF(trainData2[i, 0], trainData2[i, 1]);
                img.Draw(new CircleF(p2, 2.0f), colors[1], -1);
                PointF p3 = new PointF(trainData3[i, 0], trainData3[i, 1]);
                img.Draw(new CircleF(p3, 2.0f), colors[2], -1);
            }

            Emgu.CV.UI.ImageViewer.Show(img);
        }
        private static void Do5Clusters()
        {
            Bgr[] colors = new Bgr[] {
                new Bgr(0, 0, 255),
                new Bgr(0, 255, 0),
                new Bgr(255, 0, 0),
                new Bgr(255, 255, 0),
                new Bgr(255, 255, 255),
            };
            int trainSampleCount = 500;

            #region Generate the training data and classes
            Matrix<float> trainData = new Matrix<float>(trainSampleCount, 2);
            Matrix<int> trainClasses = new Matrix<int>(trainSampleCount, 1);

            Image<Bgr, Byte> img = new Image<Bgr, byte>(500, 500);

            Matrix<float> sample = new Matrix<float>(1, 2);

            Matrix<float> trainData1 = trainData.GetRows(0, trainSampleCount / 5, 1);
            trainData1.GetCols(0, 1).SetRandNormal(new MCvScalar(80), new MCvScalar(50));
            trainData1.GetCols(1, 2).SetRandNormal(new MCvScalar(150), new MCvScalar(50));

            Matrix<float> trainData2 = trainData.GetRows(trainSampleCount / 5, 2 * trainSampleCount / 5, 1);
            trainData2.SetRandNormal(new MCvScalar(200), new MCvScalar(50));

            Matrix<float> trainData3 = trainData.GetRows(2 * trainSampleCount / 5, 3 * trainSampleCount / 5, 1);
            trainData3.GetCols(0, 1).SetRandNormal(new MCvScalar(200), new MCvScalar(50));
            trainData3.GetCols(1, 2).SetRandNormal(new MCvScalar(400), new MCvScalar(50));

            Matrix<float> trainData4 = trainData.GetRows(3 * trainSampleCount / 5, 4 * trainSampleCount / 5, 1);
            trainData4.SetRandNormal(new MCvScalar(300), new MCvScalar(50));

            Matrix<float> trainData5 = trainData.GetRows(4 * trainSampleCount / 5, trainSampleCount, 1);
            trainData5.GetCols(0, 1).SetRandNormal(new MCvScalar(150), new MCvScalar(50));
            trainData5.GetCols(1, 2).SetRandNormal(new MCvScalar(80), new MCvScalar(50));



            Matrix<int> trainClasses1 = trainClasses.GetRows(0, trainSampleCount / 5, 1);
            trainClasses1.SetValue(1);
            Matrix<int> trainClasses2 = trainClasses.GetRows(trainSampleCount / 5, 2 * trainSampleCount / 5, 1);
            trainClasses2.SetValue(2);
            Matrix<int> trainClasses3 = trainClasses.GetRows(2 * trainSampleCount / 5, 3 * trainSampleCount / 5, 1);
            trainClasses3.SetValue(3);
            Matrix<int> trainClasses4 = trainClasses.GetRows(3 * trainSampleCount / 5, 4 * trainSampleCount / 5, 1);
            trainClasses4.SetValue(4);
            Matrix<int> trainClasses5 = trainClasses.GetRows(4 * trainSampleCount / 5, trainSampleCount, 1);
            trainClasses5.SetValue(5);
            #endregion

            using (TrainData td = new TrainData(trainData, Emgu.CV.ML.MlEnum.DataLayoutType.RowSample, trainClasses))
            using (NormalBayesClassifier classifier = new NormalBayesClassifier())
            {
                //ParamDef[] defs = classifier.GetParams();
                classifier.Train(trainData, Emgu.CV.ML.MlEnum.DataLayoutType.RowSample, trainClasses);
                classifier.Clear();
                classifier.Train(td);
#if !NETFX_CORE
                String fileName = Path.Combine(Path.GetTempPath(), "normalBayes.xml");
                classifier.Save(fileName);
                if (File.Exists(fileName))
                    File.Delete(fileName);
#endif

                #region Classify every image pixel
                for (int i = 0; i < img.Height; i++)
                    for (int j = 0; j < img.Width; j++)
                    {
                        sample.Data[0, 0] = i;
                        sample.Data[0, 1] = j;
                        int response = (int)classifier.Predict(sample, null);

                        Bgr color = colors[response - 1];

                        img[j, i] = new Bgr(color.Blue * 0.5, color.Green * 0.5, color.Red * 0.5);
                    }
                #endregion
            }

            // display the original training samples
            for (int i = 0; i < (trainSampleCount/5); i++)
            {
                PointF p1 = new PointF(trainData1[i, 0], trainData1[i, 1]);
                img.Draw(new CircleF(p1, 2.0f), colors[0], -1);
                PointF p2 = new PointF(trainData2[i, 0], trainData2[i, 1]);
                img.Draw(new CircleF(p2, 2.0f), colors[1], -1);
                PointF p3 = new PointF(trainData3[i, 0], trainData3[i, 1]);
                img.Draw(new CircleF(p3, 2.0f), colors[2], -1);
                PointF p4 = new PointF(trainData4[i, 0], trainData4[i, 1]);
                img.Draw(new CircleF(p4, 2.0f), colors[3], -1);
                PointF p5 = new PointF(trainData5[i, 0], trainData5[i, 1]);
                img.Draw(new CircleF(p5, 2.0f), colors[4], -1);
            }

            Emgu.CV.UI.ImageViewer.Show(img);
        }
    }
}
