using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

namespace TrajectoryGeneratorAPI
{
    public class Result<T>
    {
        public Result(bool success, T? value, ResultInfo info)
        {
            Success = success;
            Value = value;
            Info = info;
        }

        public Result(bool succeeded, ResultInfo info)
        {
            Success = succeeded;
            Info = info;
            Value = default(T);
        }

        public Result(bool succeeded, T value)
        {
            Success = succeeded;
            Value = value;
            Info = new ResultInfo(ServerStatus.Ok, "No errors detected");
        }

        public static Result<T> Error(Exception exception)
        {
            Console.WriteLine($"Exception: {exception.Message}\n{exception.ToString()}");
            return new Result<T>(false, new ResultInfo(exception.Message, exception));
        }

        [JsonPropertyName("success")]
        public bool Success { get; }
        [JsonPropertyName("value")]
        public T? Value { get; }
        [JsonPropertyName("info")]
        public ResultInfo Info { get; }
    }

    public class ResultInfo
    {
        public ResultInfo(ServerStatus status, string message)
        {
            Status = status;
            Message = message;
        }

        public ResultInfo(string message, Exception exception)
        {
            this.Message = message;
            this.Exception = exception;
            this.Status = ServerStatus.Exception;
        }

        [JsonPropertyName("status")]
        public ServerStatus Status { get; }
        [JsonPropertyName("message")]
        public string Message { get; }
        [JsonPropertyName("exception")]
        public Exception? Exception { get; }
    }

    public enum ServerStatus
    {
        Ok,
        NotFound,
        BadRequest,
        ServerError,
        Exception
    }
}
