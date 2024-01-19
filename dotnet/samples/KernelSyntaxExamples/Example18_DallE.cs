﻿// Copyright (c) Microsoft. All rights reserved.

using System;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Http.Resilience;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.TextToImage;
using Xunit;
using Xunit.Abstractions;

namespace Examples;

// The following example shows how to use Semantic Kernel with OpenAI Dall-E 2 to create images
public class Example18_DallE : BaseTest
{
    [Fact]
    public async Task OpenAIDallEAsync()
    {
        WriteLine("======== OpenAI Dall-E 2 Text To Image ========");

        Kernel kernel = Kernel.CreateBuilder()
            .AddOpenAITextToImage(TestConfiguration.OpenAI.ApiKey) // Add your text to image service
            .AddOpenAIChatCompletion(TestConfiguration.OpenAI.ChatModelId, TestConfiguration.OpenAI.ApiKey) // Add your chat completion service
            .Build();

        ITextToImageService dallE = kernel.GetRequiredService<ITextToImageService>();

        var imageDescription = "A cute baby sea otter";
        var image = await dallE.GenerateImageAsync(imageDescription, 256, 256);

        WriteLine(imageDescription);
        WriteLine("Image URL: " + image);

        /* Output:

        A cute baby sea otter
        Image URL: https://oaidalleapiprodscus.blob.core.windows.net/private/....

        */

        WriteLine("======== Chat with images ========");

        var chatGPT = kernel.GetRequiredService<IChatCompletionService>();
        var chatHistory = new ChatHistory(
           "You're chatting with a user. Instead of replying directly to the user" +
           " provide the description of an image that expresses what you want to say." +
           " The user won't see your message, they will see only the image. The system " +
           " generates an image using your description, so it's important you describe the image with details.");

        var msg = "Hi, I'm from Tokyo, where are you from?";
        chatHistory.AddUserMessage(msg);
        WriteLine("User: " + msg);

        var reply = await chatGPT.GetChatMessageContentAsync(chatHistory);
        chatHistory.Add(reply);
        image = await dallE.GenerateImageAsync(reply.Content!, 256, 256);
        WriteLine("Bot: " + image);
        WriteLine("Img description: " + reply);

        msg = "Oh, wow. Not sure where that is, could you provide more details?";
        chatHistory.AddUserMessage(msg);
        WriteLine("User: " + msg);

        reply = await chatGPT.GetChatMessageContentAsync(chatHistory);
        chatHistory.Add(reply);
        image = await dallE.GenerateImageAsync(reply.Content!, 256, 256);
        WriteLine("Bot: " + image);
        WriteLine("Img description: " + reply);

        /* Output:

        User: Hi, I'm from Tokyo, where are you from?
        Bot: https://oaidalleapiprodscus.blob.core.windows.net/private/...
        Img description: [An image of a globe with a pin dropped on a location in the middle of the ocean]

        User: Oh, wow. Not sure where that is, could you provide more details?
        Bot: https://oaidalleapiprodscus.blob.core.windows.net/private/...
        Img description: [An image of a map zooming in on the pin location, revealing a small island with a palm tree on it]

        */
    }

    [Fact(Skip = "Generating the Image can take too long and often break the test")]
    public async Task AzureOpenAIDallEAsync()
    {
        WriteLine("========Azure OpenAI Dall-E 3 Text To Image ========");

        var builder = Kernel.CreateBuilder()
            .AddAzureOpenAITextToImage( // Add your text to image service
                deploymentName: TestConfiguration.AzureOpenAI.ImageModelId,
                endpoint: TestConfiguration.AzureOpenAI.ImageEndpoint,
                apiKey: TestConfiguration.AzureOpenAI.ImageApiKey,
                modelId: TestConfiguration.AzureOpenAI.ImageModelId,
                apiVersion: "2023-12-01-preview") //Dall-E 3 is only supported in this version
            .AddAzureOpenAIChatCompletion( // Add your chat completion service
                deploymentName: TestConfiguration.AzureOpenAI.ChatDeploymentName,
                endpoint: TestConfiguration.AzureOpenAI.Endpoint,
                apiKey: TestConfiguration.AzureOpenAI.ApiKey);

        builder.Services.ConfigureHttpClientDefaults(c =>
        {
            // Use a standard resiliency policy, augmented to retry 5 times
            c.AddStandardResilienceHandler().Configure(o =>
            {
                o.Retry.MaxRetryAttempts = 5;
                o.TotalRequestTimeout.Timeout = TimeSpan.FromSeconds(60);
            });
        });

        var kernel = builder.Build();

        ITextToImageService dallE = kernel.GetRequiredService<ITextToImageService>();
        var imageDescription = "A cute baby sea otter";
        var image = await dallE.GenerateImageAsync(imageDescription, 1024, 1024);

        WriteLine(imageDescription);
        WriteLine("Image URL: " + image);

        /* Output:

        A cute baby sea otter
        Image URL: https://dalleproduse.blob.core.windows.net/private/images/....

        */

        WriteLine("======== Chat with images ========");

        var chatGPT = kernel.GetRequiredService<IChatCompletionService>();
        var chatHistory = new ChatHistory(
            "You're chatting with a user. Instead of replying directly to the user" +
            " provide the description of an image that expresses what you want to say." +
            " The user won't see your message, they will see only the image. The system " +
            " generates an image using your description, so it's important you describe the image with details.");

        var msg = "Hi, I'm from Tokyo, where are you from?";
        chatHistory.AddUserMessage(msg);
        WriteLine("User: " + msg);

        var reply = await chatGPT.GetChatMessageContentAsync(chatHistory);
        chatHistory.Add(reply);
        image = await dallE.GenerateImageAsync(reply.Content!, 1024, 1024);
        WriteLine("Bot: " + image);
        WriteLine("Img description: " + reply);

        msg = "Oh, wow. Not sure where that is, could you provide more details?";
        chatHistory.AddUserMessage(msg);
        WriteLine("User: " + msg);

        reply = await chatGPT.GetChatMessageContentAsync(chatHistory);
        chatHistory.Add(reply);
        image = await dallE.GenerateImageAsync(reply.Content!, 1024, 1024);
        WriteLine("Bot: " + image);
        WriteLine("Img description: " + reply);

        /* Output:

        User: Hi, I'm from Tokyo, where are you from?
        Bot: https://dalleproduse.blob.core.windows.net/private/images/......
        Img description: [An image of a globe with a pin dropped on a location in the middle of the ocean]

        User: Oh, wow. Not sure where that is, could you provide more details?
        Bot: https://dalleproduse.blob.core.windows.net/private/images/......
        Img description: [An image of a map zooming in on the pin location, revealing a small island with a palm tree on it]

        */
    }

    public Example18_DallE(ITestOutputHelper output) : base(output)
    {
    }
}
