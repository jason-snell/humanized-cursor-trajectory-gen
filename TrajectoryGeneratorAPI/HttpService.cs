using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Text;
using System.Text.Encodings.Web;
using System.Text.Json.Nodes;
using System.Text.Json;
using System.Threading.Tasks;

namespace TrajectoryGeneratorAPI
{
    public class HttpService
    {
        private readonly HttpListener listener;
        private readonly RouterService router;
        private bool isRunning;

        public HttpService(RouterService router)
        {
            this.listener = new HttpListener();
            this.router = router;
        }

        public async Task StartAsync(string[] prefixes)
        {
            try
            {
                foreach (var prefix in prefixes)
                {
                    listener.Prefixes.Add(prefix);
                }

                listener.Start();
                isRunning = true;

                Console.WriteLine(string.Join("\n", prefixes.Select(x => "Listening to " + x)));

                while (isRunning)
                {
                    var context = await listener.GetContextAsync();
                    _ = ProcessRequestAsync(context);
                }
            }
            catch (Exception ex)
            {
                isRunning = false;
                Console.WriteLine($"Failed to start server: {ex.Message}\n\n{ex.ToString()}");
            }
        }

        private async Task ProcessRequestAsync(HttpListenerContext context)
        {
            try
            {
                await router.HandleRequest(context);
            }
            catch (Exception ex)
            {
                await router.SendResponse(context.Response, Result<object>.Error(ex));
            }
        }

        public Task StopAsync()
        {
            isRunning = false;
            listener.Stop();
            return Task.CompletedTask;
        }
    }
}
