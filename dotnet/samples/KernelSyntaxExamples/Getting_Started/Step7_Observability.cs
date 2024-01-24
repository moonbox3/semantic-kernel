﻿// Copyright (c) Microsoft. All rights reserved.

using System;
using System.ComponentModel;
using System.Threading.Tasks;
using Examples;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Connectors.OpenAI;
using RepoUtils;
using Xunit;
using Xunit.Abstractions;

namespace GettingStarted;

public class Step7_Observability : BaseTest
{
    /// <summary>
    /// Shows how to observe the execution of a <see cref="KernelPlugin"/> instance with hooks.
    /// </summary>
    [Fact]
    public async Task ObservabilityWithHooksAsync()
    {
        // Create a kernel with OpenAI chat completion
        IKernelBuilder kernelBuilder = Kernel.CreateBuilder();
        kernelBuilder.AddOpenAIChatCompletion(
                modelId: TestConfiguration.OpenAI.ChatModelId,
                apiKey: TestConfiguration.OpenAI.ApiKey);

        kernelBuilder.Plugins.AddFromType<TimeInformation>();

        Kernel kernel = kernelBuilder.Build();

        // Handler which is called before a function is invoked
        void MyInvokingHandler(object? sender, FunctionInvokingEventArgs e)
        {
            WriteLine($"Invoking {e.Function.Name}");
        }

        // Handler which is called before a prompt is rendered
        void MyRenderingHandler(object? sender, PromptRenderingEventArgs e)
        {
            Console.WriteLine($"Rendering prompt for {e.Function.Name}");
        }

        // Handler which is called after a prompt is rendered
        void MyRenderedHandler(object? sender, PromptRenderedEventArgs e)
        {
            WriteLine($"Prompt sent to model: {e.RenderedPrompt}");
        }

        // Handler which is called after a function is invoked
        void MyInvokedHandler(object? sender, FunctionInvokedEventArgs e)
        {
            if (e.Result.Metadata is not null && e.Result.Metadata.ContainsKey("Usage"))
            {
                WriteLine($"Token usage: {e.Result.Metadata?["Usage"]?.AsJson()}");
            }
        }

        // Add the handlers to the kernel
        kernel.FunctionInvoking += MyInvokingHandler;
        kernel.PromptRendering += MyRenderingHandler;
        kernel.PromptRendered += MyRenderedHandler;
        kernel.FunctionInvoked += MyInvokedHandler;

        // Invoke the kernel with a prompt and allow the AI to automatically invoke functions
        OpenAIPromptExecutionSettings settings = new() { ToolCallBehavior = ToolCallBehavior.AutoInvokeKernelFunctions };
        WriteLine(await kernel.InvokePromptAsync("How many days until Christmas? Explain your thinking.", new(settings)));
    }

    /// <summary>
    /// Shows how to observe the execution of a <see cref="KernelPlugin"/> instance with filters.
    /// </summary>
    [Fact]
    public async Task ObservabilityWithFiltersAsync()
    {
        // Create a kernel with OpenAI chat completion
        IKernelBuilder kernelBuilder = Kernel.CreateBuilder();
        kernelBuilder.AddOpenAIChatCompletion(
                modelId: TestConfiguration.OpenAI.ChatModelId,
                apiKey: TestConfiguration.OpenAI.ApiKey);

        kernelBuilder.Plugins.AddFromType<TimeInformation>();

        kernelBuilder.Services.AddSingleton<IFunctionFilter, MyFunctionFilter>();
        kernelBuilder.Services.AddSingleton<IPromptFilter, MyPromptFilter>();

        Kernel kernel = kernelBuilder.Build();

        // Invoke the kernel with a prompt and allow the AI to automatically invoke functions
        OpenAIPromptExecutionSettings settings = new() { ToolCallBehavior = ToolCallBehavior.AutoInvokeKernelFunctions };
        WriteLine(await kernel.InvokePromptAsync("How many days until Christmas? Explain your thinking.", new(settings)));
    }

    /// <summary>
    /// A plugin that returns the current time.
    /// </summary>
    private sealed class TimeInformation
    {
        [KernelFunction]
        [Description("Retrieves the current time in UTC.")]
        public string GetCurrentUtcTime() => DateTime.UtcNow.ToString("R");
    }

    /// <summary>
    /// Function filter for observability.
    /// </summary>
    private sealed class MyFunctionFilter : IFunctionFilter
    {
        public void OnFunctionInvoked(FunctionInvokedContext context)
        {
            var metadata = context.Result.Metadata;

            if (metadata is not null && metadata.ContainsKey("Usage"))
            {
                Console.WriteLine($"Token usage: {metadata["Usage"]?.AsJson()}");
            }
        }

        public void OnFunctionInvoking(FunctionInvokingContext context)
        {
            Console.WriteLine($"Invoking {context.Function.Name}");
        }
    }

    /// <summary>
    /// Prompt filter for observability.
    /// </summary>
    private sealed class MyPromptFilter : IPromptFilter
    {
        public void OnPromptRendered(PromptRenderedContext context)
        {
            Console.WriteLine($"Prompt sent to model: {context.RenderedPrompt}");
        }

        public void OnPromptRendering(PromptRenderingContext context)
        {
            Console.WriteLine($"Rendering prompt for {context.Function.Name}");
        }
    }

    public Step7_Observability(ITestOutputHelper output) : base(output)
    {
    }
}
