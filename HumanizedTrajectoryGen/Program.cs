using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Diagnostics;
using System.Numerics.Tensors;
using System.Text.Json;

namespace HumanizedTrajectoryGen
{
    internal class Program
    {
        static readonly string onnxPath = "model.onnx";
        
        static void Main(string[] args)
        {
            Console.WriteLine("Enter nothing for randomized data:");

            var startPoint = Utils.ReadCoordinates("Enter start point (x,y): ");
            var endPoint = Utils.ReadCoordinates("Enter end point (x,y): ");

            var randomness = Utils.ReadDoubleInput("Enter randomness factor (optional, default: 1.5): ", 1.5);
            var density = Utils.ReadIntInput("Enter density (optional, default: 5): ", 5);

            var predictedControlPoints = Predict(startPoint, endPoint, randomness);

            if (predictedControlPoints == null)
            {
                return;
            }

            List<int[]>? finalPath;

            if (predictedControlPoints.Count >= 2)
            {
                finalPath = Utils.GenerateSmoothedAndDensedPath(predictedControlPoints, density);
            }
            else
            {
                finalPath = predictedControlPoints;
                Console.WriteLine("Warning: Not enough control points generated for smoothing. Path might be direct.");
            }

            Console.WriteLine("Trajectory:");
            var json = JsonSerializer.Serialize(finalPath, Utils.jsonSettings);

            Console.WriteLine(json);

            if (!Directory.Exists("output"))
                Directory.CreateDirectory("output");

            var xMax = finalPath?.Any() == true ? finalPath.Max(point => point[0]) : 1920;
            var yMax = finalPath?.Any() == true ? finalPath.Max(point => point[1]) : 1080;

            var outputPath = Path.Combine("output", $"output-{DateTime.Now.ToString("HH-mm-ss")}.png");
            Utils.RenderTrajectory(json, outputPath, xMax, yMax);

            if (File.Exists(outputPath))
                Process.Start("explorer.exe", "output");
        }

        static List<int[]>? Predict(int[] originalStart, int[] originalEnd, double randomnessFactor)
        {
            if (!File.Exists(onnxPath))
            {
                Console.WriteLine($"Error: ONNX model not found at '{onnxPath}'. Please ensure it's in the correct directory.");
                return null;
            }

            using var session = new InferenceSession(onnxPath);

            float largest = Math.Max(Math.Max(originalStart[0], originalStart[1]), Math.Max(originalEnd[0], originalEnd[1]));
            largest += largest / 2;

            float[] inputData = new float[4];

            inputData[0] = originalStart[0] + (float)(Utils.rand.NextDouble() * 2 * randomnessFactor - randomnessFactor);
            inputData[1] = originalStart[1] + (float)(Utils.rand.NextDouble() * 2 * randomnessFactor - randomnessFactor);

            inputData[2] = originalEnd[0] + (float)(Utils.rand.NextDouble() * 2 * randomnessFactor - randomnessFactor);
            inputData[3] = originalEnd[1] + (float)(Utils.rand.NextDouble() * 2 * randomnessFactor - randomnessFactor);

            var inputName = session.InputMetadata.Keys.First();

            for (int i = 0; i < inputData.Length; i++)
                inputData[i] /= largest;

            var inputTensor = new DenseTensor<float>(inputData, new int[] { 1, 2, 2 });

            var inputs = new NamedOnnxValue[] { NamedOnnxValue.CreateFromTensor(inputName, inputTensor) };

            var result = new List<int[]>();
            List<DisposableNamedOnnxValue>? results = null;

            try
            {
                results = session.Run(inputs)?.ToList();

                if(results == null)
                {
                    Console.WriteLine($"Error: Results are null");
                    return null;
                }

                var outputName = session.OutputMetadata.Keys.First();
                var outputTensor = results.First().AsTensor<float>();

                var predictions = outputTensor.ToArray();
                int maxDistance = 3;

                result.Add(originalStart);

                for (int i = 0; i < predictions.Length / 2; i++)
                {
                    int predictedX = (int)(predictions[i * 2] * largest);
                    int predictedY = (int)(predictions[i * 2 + 1] * largest);

                    if (Math.Abs(predictedX - originalEnd[0]) <= maxDistance &&
                        Math.Abs(predictedY - originalEnd[1]) <= maxDistance)
                    {
                        break;
                    }

                    result.Add(new int[] { predictedX, predictedY });
                }

                result.Add(originalEnd);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"An error occurred during ONNX model prediction: {ex.Message}");
                return null;
            }

            return result;
        }
    }
}
