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
using System.Xml;
using System.Xml.Linq;

namespace SVM__Support_Vector_Machine_
{
    internal class Program
    {
        static void Main(string[] args)
        {
            SVMBase();
        }
        private static void SVMBase()
        {
            int trainSampleCount = 150;
            int sigma = 60;

            #region Generate the training data and classes

            Matrix<float> trainData = new Matrix<float>(trainSampleCount, 2);
            Matrix<int> trainClasses = new Matrix<int>(trainSampleCount, 1);

            Image<Bgr, Byte> img = new Image<Bgr, byte>(500, 500);

            Matrix<float> sample = new Matrix<float>(1, 2);

            Matrix<float> trainData1 = trainData.GetRows(0, trainSampleCount / 3, 1);
            trainData1.GetCols(0, 1).SetRandNormal(new MCvScalar(100), new MCvScalar(sigma));
            trainData1.GetCols(1, 2).SetRandNormal(new MCvScalar(300), new MCvScalar(sigma));

            Matrix<float> trainData2 = trainData.GetRows(trainSampleCount / 3, 2 * trainSampleCount / 3, 1);
            trainData2.SetRandNormal(new MCvScalar(400), new MCvScalar(sigma));

            Matrix<float> trainData3 = trainData.GetRows(2 * trainSampleCount / 3, trainSampleCount, 1);
            trainData3.GetCols(0, 1).SetRandNormal(new MCvScalar(300), new MCvScalar(sigma));
            trainData3.GetCols(1, 2).SetRandNormal(new MCvScalar(100), new MCvScalar(sigma));

            Matrix<int> trainClasses1 = trainClasses.GetRows(0, trainSampleCount / 3, 1);
            trainClasses1.SetValue(1);
            Matrix<int> trainClasses2 = trainClasses.GetRows(trainSampleCount / 3, 2 * trainSampleCount / 3, 1);
            trainClasses2.SetValue(2);
            Matrix<int> trainClasses3 = trainClasses.GetRows(2 * trainSampleCount / 3, trainSampleCount, 1);
            trainClasses3.SetValue(3);

            #endregion
            try
            {
                using (SVM model = new SVM())
                {
                    
                    //SVMParams p = new SVMParams();
                    //p.KernelType = Emgu.CV.ML.MlEnum.SVM_KERNEL_TYPE.LINEAR;
                    //p.SVMType = Emgu.CV.ML.MlEnum.SVM_TYPE.C_SVC;
                    //p.C = 1;
                    //p.TermCrit = new MCvTermCriteria(100, 0.00001);
                    model.SetKernel(SVM.SvmKernelType.Linear);
                    model.Type = SVM.SvmType.CSvc;
                    model.C = 1;
                    model.TermCriteria = new MCvTermCriteria(100, 0.00001);

                    //bool trained = model.Train(trainData, trainClasses, null, null, p);
                    //bool trained = model.TrainAuto(trainData, trainClasses, null, null, p.MCvSVMParams, 5);
                    TrainData TD = new TrainData(trainData, Emgu.CV.ML.MlEnum.DataLayoutType.RowSample, trainClasses);
                    bool trained = model.TrainAuto(TD);
                    //把训练模型保存到本地
                    model.Save("mySVM.xml");
                    for (int i = 0; i < img.Height; i++)
                    {
                        for (int j = 0; j < img.Width; j++)
                        {
                            sample.Data[0, 0] = j;
                            sample.Data[0, 1] = i;

                            float response = model.Predict(sample);

                            img[i, j] =
                               response == 1 ? new Bgr(90, 0, 0) :
                               response == 2 ? new Bgr(0, 90, 0) :
                               new Bgr(0, 0, 90);
                        }
                    }
                    //获取支持向量(这里获取的是3个向量，即直线方程)
                    //Mat c = model.GetSupportVectors();
                    ////model.get
                    //Matrix<float> xx = new Matrix<float>(c.Rows, c.Cols);
                    //c.CopyTo(xx);
                    //for (int i = 0; i < c.Rows; i++)
                    //{
                    //    // The way the data is received changed as well 
                    //    byte[] b = c.GetData(i);
                    //    PointF p1 = new PointF((float)(b[0]), (float)(b[1]));
                    //    img.Draw(new CircleF(p1, 4), new Bgr(128, 128, 128), 2);
                    //}
                    //Console.WriteLine(c.Cols);
                    //Console.WriteLine(c.Rows);


                    //----获取支持向量-----
                    PointF[] sv_pfs= GetSupportPointfs();
                    foreach (PointF item in sv_pfs)
                    {
                        img.Draw(new CircleF(item, 4), new Bgr(128, 128, 128), 2);
                    }
                }

                // display the original training samples
                for (int i = 0; i < (trainSampleCount / 3); i++)
                {
                    PointF p1 = new PointF(trainData1[i, 0], trainData1[i, 1]);
                    img.Draw(new CircleF(p1, 2.0f), new Bgr(255, 100, 100), -1);
                    PointF p2 = new PointF(trainData2[i, 0], trainData2[i, 1]);
                    img.Draw(new CircleF(p2, 2.0f), new Bgr(100, 255, 100), -1);
                    PointF p3 = new PointF(trainData3[i, 0], trainData3[i, 1]);
                    img.Draw(new CircleF(p3, 2.0f), new Bgr(100, 100, 255), -1);
                }

                Emgu.CV.UI.ImageViewer.Show(img,"EMGU3.4.1 SVM");
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }
           
        }
        /// <summary>
        /// 获取支持向量的点集
        /// @Author：Heliophyte
        /// </summary>
        private static PointF[] GetSupportPointfs()
        {
            XDocument document = XDocument.Load("mySVM.xml");
            XElement root = document.Root.Element("opencv_ml_svm");
            //获取支持向量的点集数量
            int pCounts = int.Parse(root.Element("uncompressed_sv_total").Value);
            if (pCounts == 0)
            {
                return null;
            }
            PointF[] vfs = new PointF[pCounts];
            //2.13789581e+02 3.70247894e+02
            //1.92192932e+02 2.29739059e+02
            //2.65889404e+02 3.17239105e+02
            //2.58218262e+02 3.95134766e+02
            //5.11033264e+02 2.53483795e+02
            //3.28194794e+02 2.72949036e+02
            //1.15855988e+02 1.08835411e+02
            string ps = root.Element("uncompressed_support_vectors").Value;
            for (int i = 0; i < pCounts; i++)
            {
                string vi = ps.Split(new string[] { "\n " }, StringSplitOptions.RemoveEmptyEntries)[i];
                string[] pf = vi.Split(new string[] { " " }, StringSplitOptions.RemoveEmptyEntries);
                vfs[i] = new PointF(float.Parse(pf[0]),float.Parse(pf[1]));
            }
            return vfs;
        }
    }
   
}
