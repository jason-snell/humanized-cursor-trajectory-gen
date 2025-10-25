using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace TrajectoryGeneratorAPI
{
    public class RouterService
    {
        private readonly Dictionary<string, Dictionary<string, Func<RequestContext, Task>>> routes = new();
        private static readonly JsonSerializerOptions serializeOptions = new() 
        { 
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase 
        };

        public void AddRoute(string method, string path, Func<RequestContext, Task> handler)
        {
            if (path.EndsWith('/') && path.Length > 1)
                path = path.TrimEnd('/');

            if (!routes.ContainsKey(path))
            {
                routes[path] = new Dictionary<string, Func<RequestContext, Task>>();
            }

            routes[path][method.ToUpper()] = handler;
        }

        public async Task HandleRequest(HttpListenerContext context)
        {
            var path = context.Request.Url?.AbsolutePath ?? string.Empty;
            if (path.EndsWith('/') && path.Length > 1)
                path = path.TrimEnd('/');

            var method = context.Request.HttpMethod.ToUpper();
            var requestContext = new RequestContext(context);

            try
            {
                if (routes.ContainsKey(path) && routes[path].ContainsKey(method))
                {
                    await routes[path][method](requestContext);
                }
                else
                {
                    await requestContext.Write(new ErrorResponseMessage($"No route found for '{path}'"), statusCode: HttpStatusCode.NotFound);
                }
            }
            catch (Exception ex)
            {
                if (!requestContext.IsResponseSent)
                {
                    await requestContext.Write(new ErrorResponseMessage("Unhandled Exception: " + ex.Message), statusCode: HttpStatusCode.InternalServerError);
                }
            }
        }

        public async Task SendResponse(HttpListenerResponse response, Result<object> result)
        {
            response.ContentType = "application/json";

            try
            {
                using var memoryStream = new MemoryStream();
                using var writer = new Utf8JsonWriter(memoryStream);
                JsonSerializer.Serialize(writer, result, serializeOptions);
                await writer.FlushAsync();
                memoryStream.Position = 0;

                await memoryStream.CopyToAsync(response.OutputStream);
            }
            finally
            {
                response.Close();
            }
        }
    }

    public class ErrorResponseMessage(string message)
    {
        public bool Error = true;
        public string? Message { get; set; } = message;
    }
}
