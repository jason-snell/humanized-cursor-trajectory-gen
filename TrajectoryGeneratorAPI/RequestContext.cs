using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace TrajectoryGeneratorAPI
{
    public class RequestContext
    {
        private readonly HttpListenerRequest Request;
        private readonly HttpListenerResponse Response;

        public bool IsResponseSent { get; private set; }

        private static readonly JsonSerializerOptions serializeOptions = new()
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
        };


        public RequestContext(HttpListenerContext context)
        {
            this.Request = context.Request;
            this.Response = context.Response;
        }

        public string? GetQueryParameter(string name)
        {
            return Request.QueryString[name];
        }

        public async Task<T?> GetBodyAsync<T>()
        {
            if (!Request.HasEntityBody)
            {
                return default;
            }

            try
            {
                return await JsonSerializer.DeserializeAsync<T>(Request.InputStream, serializeOptions);
            }
            catch (JsonException)
            {
                return default;
            }
        }

        public async Task Write(object data, string contentType = "application/json", HttpStatusCode statusCode = HttpStatusCode.OK)
        {
            if (this.IsResponseSent)
                return;

            this.IsResponseSent = true;
            Response.StatusCode = (int)statusCode;
            Response.ContentType = "application/json";

            try
            {
                await JsonSerializer.SerializeAsync(Response.OutputStream, data, serializeOptions);
            }
            finally
            {
                Response.OutputStream.Close();
            }
        }
    }
}
