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
/// An expectation-maximization (EM) algorithm is used in statistics for finding maximum likelihood estimates of parameters in probabilistic models, 
/// where the model depends on unobserved latent variables. 
/// EM is an iterative method which alternates between performing an expectation (E) step,
/// which computes an expectation of the log likelihood with respect to the current estimate of the distribution for the latent variables, 
/// and a maximization (M) step, which computes the parameters which maximize the expected log likelihood found on the E step. 
/// These parameters are then used to determine the distribution of the latent variables in the next E step.
/// </summary>
namespace Expectation_Maximization
{
    internal class Program
    {
        static void Main(string[] args)
        {
            EmBase();
        }
        /// <summary>
        /// 机器学习之EM算法，基础
        /// </summary>
        private static void EmBase()
        {
            int N = 4; //number of clusters
            int N1 = (int)Math.Sqrt((double)4);
            //定义四种颜色，每一类用一种颜色表示
            Bgr[] colors = new Bgr[] {
                    new Bgr(0, 0, 255),
                    new Bgr(0, 255, 0),
                    new Bgr(255, 255, 0),
                    new Bgr(255, 0, 255)};

            int nSamples = 100;//100个样本点

            Matrix<float> samples = new Matrix<float>(nSamples, 2);//样本矩阵,100行2列，即100个坐标点 
            Matrix<Int32> labels = new Matrix<int>(nSamples, 1); //标注结果，不需要事先知道
            Image<Bgr, Byte> img = new Image<Bgr, byte>(500, 500);//待测数据，每一个坐标点为一个待测数据
            Matrix<float> sample = new Matrix<float>(1, 2);
            Matrix<float> means0 = new Matrix<float>(N, 2);//储存初始化均值
            Matrix<float> probs0 = new Matrix<float>(nSamples, 1); //输出一个矩阵，里面包含每个隐性变量的后验概率
            CvInvoke.cvReshape(samples.Ptr, samples.Ptr, 2, 0);
            //循环生成四个类别样本数据，共样本100个，每类样本25个
            for (int i = 0; i < N; i++)
            {
                Matrix<float> rows = samples.GetRows(i * nSamples / N, (i + 1) * nSamples / N, 1);
                double scaleX = ((i % N1) + 1.0) / (N1 + 1);
                double scaleY = ((i / N1) + 1.0) / (N1 + 1);
                //设置均值
                MCvScalar mean = new MCvScalar(scaleX * img.Width, scaleY * img.Height);
                Console.WriteLine($"mean = {mean.V0}");
                //设置标准差
                MCvScalar sigma = new MCvScalar(30, 30);
                Console.WriteLine($"sigma = {sigma.V0}");
                ulong seed = (ulong)DateTime.Now.Ticks;
                Console.WriteLine($"seed = {seed}");
                //根据均值和标准差，随机生成25个正态分布坐标点作为样本
                CvInvoke.Randn(rows, mean, sigma);
            }
            CvInvoke.cvReshape(samples.Ptr, samples.Ptr, 1, 0);

            //using (EM emModel1 = new EM())
            using (EM emModel1 = new EM())
            {
                emModel1.ClustersNumber = 4;
                emModel1.CovarianceMatrixType = EM.CovarianMatrixType.Spherical;
                emModel1.TermCriteria = new MCvTermCriteria(300, 0.000001);
                //emModel1.Train(samples,Emgu.CV.ML.MlEnum.DataLayoutType.RowSample, labels);
                emModel1.trainE(samples, means0, null,null,null,labels,probs0);
                emModel1.TrainM(samples, probs0, null, labels, probs0);
                #region Classify every image pixel
                for (int i = 0; i < img.Height; i++)
                    for (int j = 0; j < img.Width; j++)
                    {
                        sample.Data[0, 0] = i;
                        sample.Data[0, 1] = j;
                        MCvPoint2D64f mCvPoint2D64F = emModel1.Predict(sample, null);
                        //这里做测试，看预测结果的分类
                        //Console.WriteLine($"{j},{i}|{mCvPoint2D64F.X},{mCvPoint2D64F.Y}");
                        int response = (int)(mCvPoint2D64F.X);
                        //Console.WriteLine($"response = {response}");
                        Bgr color = colors[response];

                        img.Draw(
                           new CircleF(new PointF(i, j), 1),
                           new Bgr(color.Blue * 0.5, color.Green * 0.5, color.Red * 0.5),
                           //color,
                           0);
                    }
                #endregion

                #region draw the clustered samples
                for (int i = 0; i < nSamples; i++)
                {
                    img.Draw(new CircleF(new PointF(samples.Data[i, 0], samples.Data[i, 1]), 1), colors[labels.Data[i, 0]], 0);
                }
                #endregion
                Emgu.CV.UI.ImageViewer.Show(img,"EM Image Result");
            }
        }
    }
}
