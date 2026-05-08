using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;
using System.Diagnostics.Metrics;
using System.Net;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace TrajectoryGeneratorAPI
{
    internal class Program {
        private static readonly string modelPath = Path.Combine("data", "model.onnx");
        private static readonly Random random = new Random();
        private static InferenceSession? session = null;
        private static string inputName = string.Empty;

        static async Task Main(string[] args) {
            if(!Directory.Exists("data")) {
                Directory.CreateDirectory("data");
                Console.WriteLine("'data' folder has been created. Download the onnx file from 'https://github.com/jason-snell/humanized-cursor-trajectory-gen' and place the file in there, named 'model.onnx'");
                Console.ReadKey();
                return;
            }

            if(!File.Exists(modelPath)) {
                Console.WriteLine($"'{modelPath}' not found. Download the onnx file from 'https://github.com/jason-snell/humanized-cursor-trajectory-gen' and place the file in there, named 'model.onnx'");
                Console.ReadKey();
                return;
            }

            if(session == null) {
                session = new InferenceSession(modelPath);
            }

            var router = new RouterService();
            var httpService = new HttpService(router);

            router.AddRoute("POST", "/generate", async (context) => {
                try
                {
                    var request = await context.GetBodyAsync<GenerateRequestMessage>();
                    if(request == null) {
                        await context.Write(new ErrorResponseMessage("Invalid request data."), statusCode: HttpStatusCode.BadRequest);
                        return;
                    }

                    if(request.Start == null || request.End == null) {
                        await context.Write(new ErrorResponseMessage($"Invalid request data. Request must include a 'start' and 'end' integer array."), statusCode: HttpStatusCode.BadRequest);
                        return;
                    }

                    if(request.Start.Length != 2) {
                        await context.Write(new ErrorResponseMessage($"Invalid request data. The expected length of 'start' is 2."));
                        return;
                    }

                    if (request.End.Length != 2) {
                        await context.Write(new ErrorResponseMessage($"Invalid request data. The expected length of 'end' is 2."));
                        return;
                    }

                    if (request.Points.HasValue) {
                        request.Points = Math.Clamp(request.Points.Value, 5, 500);
                    }

                    request.Randomness = Math.Clamp(request.Randomness, 0, 250);

                    var result = Predict(request.Start, request.End, request.Randomness, request.Points);
                    if(result == null) {
                        await context.Write(new ErrorResponseMessage("Failed to generate trajectory."), statusCode: HttpStatusCode.BadRequest);
                        return;
                    }

                    await context.Write(new {
                        points = result
                    });
                }
                catch (Exception ex) {
                    await context.Write(new ErrorResponseMessage("Unhandled Exception" + ex.Message));
                    Console.WriteLine($"[/generate] Unhandled Exception: {ex.Message}", ex.ToString());
                    return;
                }
            });

            await httpService.StartAsync(new[] { "http://127.0.0.1:42069/" });
        }

        static List<int[]>? Predict(int[] originalStart, int[] originalEnd, double randomnessFactor, int? numPoints = null) {
            if(session == null || session.InputMetadata == null || session.InputMetadata.Keys == null) {
                Console.WriteLine($"[Predict] session is null: {(session == null).ToString()}, session.InputMetadata is null: {(session?.InputMetadata == null).ToString()}, session.InputMetadata.Keys is null: {(session?.InputMetadata?.Keys == null).ToString()}");
                return null;
            }

            if (originalStart.Length != 2) {
                Console.WriteLine($"[Predict] Invalid length for originalStart. Expected 2, received {originalStart.Length}");
                return null;
            }

            if(originalEnd.Length != 2) {
                Console.WriteLine($"[Predict] Invalid length for originalEnd. Expected 2, received {originalEnd.Length}");
                return null;
            }

            float randomizedStartX = originalStart[0] + (float)(random.NextDouble() * 2 * randomnessFactor - randomnessFactor);
            float randomizedStartY = originalStart[1] + (float)(random.NextDouble() * 2 * randomnessFactor - randomnessFactor);
            float randomizedEndX = originalEnd[0] + (float)(random.NextDouble() * 2 * randomnessFactor - randomnessFactor);
            float randomizedEndY = originalEnd[1] + (float)(random.NextDouble() * 2 * randomnessFactor - randomnessFactor);

            float largest = Math.Max(Math.Max(randomizedStartX, randomizedStartY), Math.Max(randomizedEndX, randomizedEndY));
            if (largest <= 0) {
                Console.WriteLine("[Predict] All coordinates are zero or negative");
                return null;
            }

            float[] inputData = new float[4];
            inputData[0] = randomizedStartX / largest;
            inputData[1] = randomizedStartY / largest;
            inputData[2] = randomizedEndX / largest;
            inputData[3] = randomizedEndY / largest;

            if(string.IsNullOrEmpty(inputName)) {
                var _inputName = session.InputMetadata.Keys.FirstOrDefault();
                if(string.IsNullOrEmpty(_inputName)) {
                    Console.WriteLine("[Predict] Unexpected error. session.InputMetadata.Keys[0] is null.");
                    return null;
                }

                inputName = _inputName;
            }

            var inputTensor = new DenseTensor<float>(inputData, new int[] { 1, 2, 2 });
            var inputs = new NamedOnnxValue[] { NamedOnnxValue.CreateFromTensor(inputName, inputTensor) };

            var result = new List<int[]>();

            try {
                var results = session.Run(inputs)?.ToList();

                if (results == null) {
                    Console.WriteLine($"Error: Results are null");
                    return null;
                }

                var outputName = session.OutputMetadata.Keys.First();
                var outputTensor = results.First().AsTensor<float>();
                var predictions = outputTensor.ToArray();

                var generatedPoints = new List<int[]>();
                generatedPoints.Add(new int[] { (int)randomizedStartX, (int)randomizedStartY });

                for (int i = 0; i < predictions.Length / 2; i++) {
                    int predictedX = (int)(predictions[i * 2] * largest);
                    int predictedY = (int)(predictions[i * 2 + 1] * largest);
                    generatedPoints.Add(new int[] { predictedX, predictedY });
                }

                var pointsToProcess = new List<int[]>();
                if (numPoints.HasValue && numPoints.Value > 1 && generatedPoints.Count > 1) {
                    int n = numPoints ?? generatedPoints.Count;
                    result.Capacity = n;

                    int m = generatedPoints.Count;
                    var first = generatedPoints[0];
                    var last = generatedPoints[m - 1];

                    float startErrX = originalStart[0] - first[0];
                    float startErrY = originalStart[1] - first[1];
                    float endErrX = originalEnd[0] - last[0];
                    float endErrY = originalEnd[1] - last[1];

                    for (int i = 0; i < n; i++) {
                        float t = n == 1 ? 0f : (float)i / (n - 1);

                        double srcIdx = (double)i * (m - 1) / Math.Max(n - 1, 1);

                        int i1 = (int)Math.Floor(srcIdx);
                        int i2 = Math.Min(i1 + 1, m - 1);

                        double frac = srcIdx - i1;

                        var p1 = generatedPoints[i1];
                        var p2 = generatedPoints[i2];

                        double x = p1[0] + frac * (p2[0] - p1[0]) + startErrX * (1 - t) + endErrX * t;
                        double y = p1[1] + frac * (p2[1] - p1[1]) + startErrY * (1 - t) + endErrY * t;

                        result.Add(new[] { (int)Math.Round(x), (int)Math.Round(y) });
                    }
                }
                else {
                    pointsToProcess.AddRange(generatedPoints);
                }

                if (pointsToProcess.Count > 1) {
                    var actualStart = pointsToProcess.First();
                    var actualEnd = pointsToProcess.Last();

                    var desiredStart = originalStart;
                    var desiredEnd = originalEnd;

                    float startErrorX = desiredStart[0] - actualStart[0];
                    float startErrorY = desiredStart[1] - actualStart[1];
                    float endErrorX = desiredEnd[0] - actualEnd[0];
                    float endErrorY = desiredEnd[1] - actualEnd[1];

                    for (int i = 0; i < pointsToProcess.Count; i++) {
                        float t = (float)i / (pointsToProcess.Count - 1);

                        float correctionX = startErrorX * (1 - t) + endErrorX * t;
                        float correctionY = startErrorY * (1 - t) + endErrorY * t;

                        int correctedX = (int)Math.Round(pointsToProcess[i][0] + correctionX);
                        int correctedY = (int)Math.Round(pointsToProcess[i][1] + correctionY);

                        result.Add(new int[] { correctedX, correctedY });
                    }
                }
                else if (pointsToProcess.Any()) {
                    result.Add(originalEnd);
                }
            }
            catch (Exception ex) {
                Console.WriteLine($"An error occurred during ONNX model prediction: {ex.Message}");
                return null;
            }

            return result;
        }
    }

    public class GenerateRequestMessage {
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
