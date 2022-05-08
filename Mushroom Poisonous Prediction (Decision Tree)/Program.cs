using Emgu.CV;
using Emgu.CV.ML;
using Emgu.CV.Structure;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;


namespace Mushroom_Poisonous_Prediction__Decision_Tree_
{
    internal class Program
    {
        static void Main(string[] args)
        {
            DecisionTreeBase();
        }
        /// <summary>
        /// 读取本地毒蘑菇数据
        /// </summary>
        /// <param name="data"></param>
        /// <param name="response"></param>
        private static void ReadMushroomData(out Matrix<float> data, out Matrix<float> response)
        {
            string[] rows = System.IO.File.ReadAllLines("agaricus-lepiota.data");

            int varCount = rows[0].Split(',').Length - 1;
            data = new Matrix<float>(rows.Length, varCount);
            response = new Matrix<float>(rows.Length, 1);
            int count = 0;
            foreach (string row in rows)
            {
                string[] values = row.Split(',');
                Char c = System.Convert.ToChar(values[0]);
                response[count, 0] = System.Convert.ToInt32(c);
                for (int i = 1; i < values.Length; i++)
                    data[count, i - 1] = System.Convert.ToByte(System.Convert.ToChar(values[i]));
                count++;
            }
        }
        /// <summary>
        /// 决策树基础
        /// 在本例中，我们试图训练一棵决策树来识别有毒蘑菇。
        /// </summary>
        private static void DecisionTreeBase()
        {
            Matrix<float> data, response;
            ReadMushroomData(out data, out response);

            //Use the first 80% of data as training sample
            int trainingSampleCount = (int)(data.Rows * 0.8);

            Matrix<Byte> varType = new Matrix<byte>(data.Cols + 1, 1);
            varType.SetValue((byte)Emgu.CV.ML.MlEnum.VarType.Categorical); //the data is categorical

            Matrix<byte> sampleIdx = new Matrix<byte>(data.Rows, 1);
            using (Matrix<byte> sampleRows = sampleIdx.GetRows(0, trainingSampleCount, 1))
                sampleRows.SetValue(255);
            //惩罚因子
            float[] priors = new float[] { 1, 0.5f };
            GCHandle priorsHandle = GCHandle.Alloc(priors, GCHandleType.Pinned);

            //MCvDTreeParams param = new MCvDTreeParams();
            //param.maxDepth = 8;
            //param.minSampleCount = 10;
            //param.regressionAccuracy = 0;
            //param.useSurrogates = true;
            //param.maxCategories = 15;
            //param.cvFolds = 10;
            //param.use1seRule = true;
            //param.truncatePrunedTree = true;
            //param.priors = priorsHandle.AddrOfPinnedObject();

            using (DTrees dtrees = new DTrees())
            {
                dtrees.MaxDepth = 8;
                dtrees.MinSampleCount = 10;
                dtrees.RegressionAccuracy = 0;
                dtrees.MaxCategories = 15;
                dtrees.UseSurrogates = true;
                dtrees.CVFolds = 1;
                dtrees.Use1SERule = true;
                dtrees.TruncatePrunedTree = true;
                bool success = dtrees.Train(
                   data,
                   Emgu.CV.ML.MlEnum.DataLayoutType.RowSample,
                   response);

                if (!success) return;
                double trainDataCorrectRatio = 0;
                double testDataCorrectRatio = 0;
                for (int i = 0; i < data.Rows; i++)
                {
                    using (Matrix<float> sample = data.GetRow(i))
                    {
                        double r = dtrees.Predict(sample);
                        r = Math.Abs(r - response[i, 0]);
                        if (r < 1.0e-5)
                        {
                            if (i < trainingSampleCount)
                                trainDataCorrectRatio++;
                            else
                                testDataCorrectRatio++;
                        }
                    }
                }

                trainDataCorrectRatio /= trainingSampleCount;
                testDataCorrectRatio /= (data.Rows - trainingSampleCount);

                Trace.WriteLine(String.Format("Prediction accuracy for training data :{0}%", trainDataCorrectRatio * 100));
                Trace.WriteLine(String.Format("Prediction accuracy for test data :{0}%", testDataCorrectRatio * 100));
            }

            priorsHandle.Free();
        }
    }
}
