using SkiaSharp;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace HumanizedTrajectoryGen
{
    public class Utils
    {
        public static Random rand = new Random();

        public static JsonSerializerOptions jsonSettings = new JsonSerializerOptions()
        {
            DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull,
            WriteIndented = false
        };

        private static SKPaint skWhite = new SKPaint()
        {
            Color = SKColors.White,
            IsAntialias = true,
            StrokeWidth = 1
        };

        private static SKPaint skRed = new SKPaint()
        {
            Color = SKColors.Red,
            IsAntialias = true
        };

        public static int[] ReadCoordinates(string prompt)
        {
            int[] coords = new int[2];
            while (true)
            {
                Console.Write(prompt);
                string? input = Console.ReadLine();

                if (string.IsNullOrWhiteSpace(input))
                {
                    coords[0] = rand.Next(0, 1920);
                    coords[1] = rand.Next(0, 1080);

                    Console.WriteLine($"No input provided. Generating random coordinates: {coords[0]},{coords[1]}");
                    return coords;
                }

                var parts = input.Split(',').Select(p => p.Trim()).ToArray();

                if (parts.Length != 2 || !int.TryParse(parts[0], out coords[0]) || !int.TryParse(parts[1], out coords[1]))
                {
                    Console.WriteLine("Invalid format. Please enter two numbers separated by a comma (e.g., 100,200).");
                    continue;
                }

                return coords;
            }
        }

        public static int ReadIntInput(string prompt, int defaultValue)
        {
            while (true)
            {
                Console.Write($"{prompt} (Default: {defaultValue}) ");
                string? input = Console.ReadLine();

                if (string.IsNullOrWhiteSpace(input) || 
                    !int.TryParse(input, out int value) ||
                    value < 1)
                {
                    Console.WriteLine("Value must be at least 1. Using default.");
                    return defaultValue;
                }

                return value;
            }
        }

        public static double ReadDoubleInput(string prompt, double defaultValue)
        {
            while (true)
            {
                Console.Write($"{prompt} (Default: {defaultValue}) ");
                string? input = Console.ReadLine();

                if (string.IsNullOrWhiteSpace(input) || 
                    !double.TryParse(input, out double value) || 
                    value < 0) 
                {
                    Console.WriteLine("Invalid input. Please enter a number.");
                    return defaultValue;
                }

                return value;
            }
        }

        public static double[] CatmullRom(double p0x, double p0y, double p1x, double p1y, double p2x, double p2y, double p3x, double p3y, double t)
        {
            var t2 = t * t;
            var t3 = t2 * t;

            var x = 0.5 * ((2 * p1x) + (-p0x + p2x) * t +
                              (2 * p0x - 5 * p1x + 4 * p2x - p3x) * t2 +
                              (-p0x + 3 * p1x - 3 * p2x + p3x) * t3);

            var y = 0.5 * ((2 * p1y) + (-p0y + p2y) * t +
                              (2 * p0y - 5 * p1y + 4 * p2y - p3y) * t2 +
                              (-p0y + 3 * p1y - 3 * p2y + p3y) * t3);

            return new double[] { x, y };
        }

        public static List<int[]>? GenerateSmoothedAndDensedPath(List<int[]> controlPoints, int pointsPerSegment)
        {
            if (controlPoints == null || controlPoints.Count < 2)
            {
                return null;
            }

            var interpolatedPath = new List<int[]>();

            interpolatedPath.Add(controlPoints[0]);

            for (int i = 0; i < controlPoints.Count - 1; i++)
            {
                var p0x = controlPoints[Math.Max(0, i - 1)][0];
                var p0y = controlPoints[Math.Max(0, i - 1)][1];
                
                var p1x = controlPoints[i][0];
                var p1y = controlPoints[i][1];
                
                var p2x = controlPoints[i + 1][0];
                var p2y = controlPoints[i + 1][1];
                
                var p3x = controlPoints[Math.Min(controlPoints.Count - 1, i + 2)][0];
                var p3y = controlPoints[Math.Min(controlPoints.Count - 1, i + 2)][1];

                for (int j = 1; j <= pointsPerSegment; j++)
                {
                    var t = (double)j / pointsPerSegment;
                    double[] interpolatedPoint = CatmullRom(p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y, t);
                    interpolatedPath.Add(new int[] { (int)Math.Round(interpolatedPoint[0]), (int)Math.Round(interpolatedPoint[1]) });
                }
            }

            return interpolatedPath;
        }

        public static void RenderTrajectory(string jsonCoordinates, string outputPath, int imageWidth = 800, int imageHeight = 1300)
        {
            try
            {
                var coordinates = JsonSerializer.Deserialize<int[][]>(jsonCoordinates);
                if (coordinates == null || coordinates.Length == 0)
                {
                    Console.WriteLine("No coordinates provided or failed to deserialize input: " + jsonCoordinates);
                    return;
                }

                int canvasWidth = 1000;
                int canvasHeight = 1000;
                float padding = 50f;

                using var bitmap = new SKBitmap(canvasWidth, canvasHeight);
                using var canvas = new SKCanvas(bitmap);

                canvas.Clear(SKColors.Black);

                float minX = float.MaxValue, minY = float.MaxValue;
                float maxX = float.MinValue, maxY = float.MinValue;

                var firstPoint = true;
                foreach (var point in coordinates)
                {
                    if (point == null || point.Length < 2) continue;

                    float x = point[0];
                    float y = point[1];

                    if (firstPoint)
                    {
                        minX = x;
                        minY = y;
                        maxX = x;
                        maxY = y;
                        firstPoint = false;
                    }
                    else
                    {
                        minX = Math.Min(minX, x);
                        minY = Math.Min(minY, y);
                        maxX = Math.Max(maxX, x);
                        maxY = Math.Max(maxY, y);
                    }
                }

                if (firstPoint)
                {
                    Console.WriteLine("No valid coordinate points found to render.");
                    return;
                }

                var dataWidth = maxX - minX;
                var dataHeight = maxY - minY;

                var actualScale = 1f;
                var targetDrawingWidth = canvasWidth - 2 * padding;
                var targetDrawingHeight = canvasHeight - 2 * padding;

                if (dataWidth > 0 && dataHeight > 0)
                {
                    actualScale = Math.Min(targetDrawingWidth / dataWidth, targetDrawingHeight / dataHeight);
                }
                else if (dataWidth > 0)
                {
                    actualScale = targetDrawingWidth / dataWidth;
                }
                else if (dataHeight > 0)
                {
                    actualScale = targetDrawingHeight / dataHeight;
                }

                var dataCenterX = (minX + maxX) / 2f;
                var dataCenterY = (minY + maxY) / 2f;

                canvas.Translate(canvasWidth / 2f, canvasHeight / 2f);
                canvas.Scale(actualScale);
                canvas.Translate(-dataCenterX, -dataCenterY);

                for (int i = 0; i < coordinates.Length - 1; i++)
                {
                    if (coordinates[i] == null || coordinates[i].Length < 2 ||
                        coordinates[i + 1] == null || coordinates[i + 1].Length < 2)
                    {
                        continue;
                    }

                    var x1 = coordinates[i][0];
                    var y1 = coordinates[i][1];
                    var x2 = coordinates[i + 1][0];
                    var y2 = coordinates[i + 1][1];

                    canvas.DrawLine(x1, y1, x2, y2, skWhite);

                    if(i % 4 == 0)
                        canvas.DrawCircle(x1, y1, 2.5f, skRed);
                }

                using var image = SKImage.FromBitmap(bitmap);
                using var data = image.Encode(SKEncodedImageFormat.Png, 100);
                using var stream = File.OpenWrite(outputPath);

                data.SaveTo(stream);

                Console.WriteLine($"Image saved successfully to {outputPath}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
            }
        }
    }
}
