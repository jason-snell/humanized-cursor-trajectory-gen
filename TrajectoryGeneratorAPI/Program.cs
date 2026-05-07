using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;
using System.Diagnostics.Metrics;
using System.Net;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace TrajectoryGeneratorAPI
{
    internal class Program
    {
        private static readonly string modelPath = Path.Combine("data", "model.onnx");
        private static readonly Random random = new Random();
        private static InferenceSession? session = null;

        static async Task Main(string[] args)
        {
            if(!Directory.Exists("data"))
            {
                Directory.CreateDirectory("data");
                Console.WriteLine("'data' folder has been created. Download the onnx file from 'https://github.com/jason-snell/humanized-cursor-trajectory-gen' and place the file in there, named 'model.onnx'");
                Console.ReadKey();
                return;
            }

            if(!File.Exists(modelPath))
            {
                Console.WriteLine($"'{modelPath}' not found. Download the onnx file from 'https://github.com/jason-snell/humanized-cursor-trajectory-gen' and place the file in there, named 'model.onnx'");
                Console.ReadKey();
                return;
            }

            if(session == null)
            {
                session = new InferenceSession(modelPath);
            }

            var router = new RouterService();
            var httpService = new HttpService(router);

            router.AddRoute("POST", "/generate", async (context) => {
                try
                {
                    var request = await context.GetBodyAsync<GenerateRequestMessage>();
                    if(request == null)
                    {
                        await context.Write(new ErrorResponseMessage("Invalid request data."), statusCode: HttpStatusCode.BadRequest);
                        return;
                    }

                    if(request.Start == null || request.End == null)
                    {
                        await context.Write(new ErrorResponseMessage($"Invalid request data. Request must include a 'start' and 'end' integer array."), statusCode: HttpStatusCode.BadRequest);
                        return;
                    }

                    if (!TryValidatePoint(request.Start, nameof(request.Start), out var pointError) ||
                        !TryValidatePoint(request.End, nameof(request.End), out pointError))
                    {
                        await context.Write(new ErrorResponseMessage(pointError!), statusCode: HttpStatusCode.BadRequest);
                        return;
                    }

                    if(request.Points.HasValue)
                    {
                        request.Points = Math.Clamp(request.Points.Value, 5, 500);
                    }

                    request.Randomness = Math.Clamp(request.Randomness, 0, 250);

                    var result = Predict(request.Start, request.End, request.Randomness, request.Points);
                    if(result == null)
                    {
                        await context.Write(new ErrorResponseMessage("Failed to generate trajectory."), statusCode: HttpStatusCode.BadRequest);
                        return;
                    }

                    await context.Write(new
                    {
                        points = result
                    });
                }
                catch (Exception ex)
                {
                    await context.Write(new ErrorResponseMessage("Unhandled Exception" + ex.Message));
                    Console.WriteLine($"[/generate] Unhandled Exception: {ex.Message}", ex.ToString());
                    return;
                }
            });

            await httpService.StartAsync(new[] { "http://127.0.0.1:42069/" });
        }

        static List<int[]>? Predict(int[] originalStart, int[] originalEnd, double randomnessFactor, int? numPoints = null)
        {
            if (!TryValidatePoint(originalStart, nameof(originalStart), out var startError) ||
                !TryValidatePoint(originalEnd, nameof(originalEnd), out var endError))
            {
                Console.WriteLine(startError ?? endError);
                return null;
            }

            if (session == null)
            {
                Console.WriteLine($"model not loaded");
                return null;
            }

            float randomizedStartX = originalStart[0] + (float)(random.NextDouble() * 2 * randomnessFactor - randomnessFactor);
            float randomizedStartY = originalStart[1] + (float)(random.NextDouble() * 2 * randomnessFactor - randomnessFactor);
            float randomizedEndX = originalEnd[0] + (float)(random.NextDouble() * 2 * randomnessFactor - randomnessFactor);
            float randomizedEndY = originalEnd[1] + (float)(random.NextDouble() * 2 * randomnessFactor - randomnessFactor);

            float[] inputData = new float[4];
            inputData[0] = randomizedStartX;
            inputData[1] = randomizedStartY;
            inputData[2] = randomizedEndX;
            inputData[3] = randomizedEndY;

            float largest = GetNormalizationScale(inputData);

            inputData[0] /= largest;
            inputData[1] /= largest;
            inputData[2] /= largest;
            inputData[3] /= largest;

            var inputName = session.InputMetadata.Keys.First();
            var inputTensor = new DenseTensor<float>(inputData, new int[] { 1, 2, 2 });
            var inputs = new NamedOnnxValue[] { NamedOnnxValue.CreateFromTensor(inputName, inputTensor) };

            var result = new List<int[]>();

            try
            {
                var results = session.Run(inputs)?.ToList();

                if (results == null)
                {
                    Console.WriteLine($"Error: Results are null");
                    return null;
                }

                var outputName = session.OutputMetadata.Keys.First();
                var outputTensor = results.First().AsTensor<float>();
                var predictions = outputTensor.ToArray();

                var generatedPoints = new List<int[]>();
                generatedPoints.Add(new int[] { (int)randomizedStartX, (int)randomizedStartY });

                for (int i = 0; i < predictions.Length / 2; i++)
                {
                    int predictedX = (int)(predictions[i * 2] * largest);
                    int predictedY = (int)(predictions[i * 2 + 1] * largest);
                    generatedPoints.Add(new int[] { predictedX, predictedY });
                }

                var pointsToProcess = new List<int[]>();
                if (numPoints.HasValue && numPoints.Value > 1 && generatedPoints.Count > 1)
                {
                    int n = numPoints.Value;
                    int m = generatedPoints.Count;

                    for (int i = 0; i < n; i++)
                    {
                        double sourceIndex = (double)i * (m - 1) / (n - 1);
                        int index1 = (int)Math.Floor(sourceIndex);
                        int index2 = (int)Math.Ceiling(sourceIndex);
                        double t = sourceIndex - index1;

                        if (index1 == index2)
                        {
                            pointsToProcess.Add(generatedPoints[index1]);
                        }
                        else
                        {
                            int[] p1 = generatedPoints[index1];
                            int[] p2 = generatedPoints[index2];
                            int newX = (int)Math.Round(p1[0] + t * (p2[0] - p1[0]));
                            int newY = (int)Math.Round(p1[1] + t * (p2[1] - p1[1]));
                            pointsToProcess.Add(new int[] { newX, newY });
                        }
                    }
                }
                else
                {
                    pointsToProcess.AddRange(generatedPoints);
                }

                if (pointsToProcess.Count > 1)
                {
                    var actualStart = pointsToProcess.First();
                    var actualEnd = pointsToProcess.Last();

                    var desiredStart = originalStart;
                    var desiredEnd = originalEnd;

                    float startErrorX = desiredStart[0] - actualStart[0];
                    float startErrorY = desiredStart[1] - actualStart[1];
                    float endErrorX = desiredEnd[0] - actualEnd[0];
                    float endErrorY = desiredEnd[1] - actualEnd[1];

                    for (int i = 0; i < pointsToProcess.Count; i++)
                    {
                        float t = (float)i / (pointsToProcess.Count - 1);

                        float correctionX = startErrorX * (1 - t) + endErrorX * t;
                        float correctionY = startErrorY * (1 - t) + endErrorY * t;

                        int correctedX = (int)Math.Round(pointsToProcess[i][0] + correctionX);
                        int correctedY = (int)Math.Round(pointsToProcess[i][1] + correctionY);

                        result.Add(new int[] { correctedX, correctedY });
                    }
                }
                else if (pointsToProcess.Any())
                {
                    result.Add(originalEnd);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"An error occurred during ONNX model prediction: {ex.Message}");
                return null;
            }

            return result;
        }

        static bool TryValidatePoint(int[]? point, string label, out string? error)
        {
            if (point == null || point.Length != 2)
            {
                error = $"Invalid request data. {label} must contain exactly two integer values.";
                return false;
            }

            error = null;
            return true;
        }

        static float GetNormalizationScale(IEnumerable<float> values)
        {
            var largestMagnitude = values.Select(Math.Abs).DefaultIfEmpty(0f).Max();
            if (largestMagnitude < 1f)
            {
                return 1f;
            }

            return largestMagnitude * 1.5f;
        }
    }

    public class GenerateRequestMessage
    {
        [JsonPropertyName("start")]
        public int[]? Start { get; set; }
        [JsonPropertyName("end")]
        public int[]? End { get; set; }
        [JsonPropertyName("points")]
        public int? Points { get; set; }
        [JsonPropertyName("randomness")]
        public float Randomness { get; set; } = 1;
    }
}
