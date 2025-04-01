### SK Function Call Handling Summary

#### Chat Completion Base Layer: `ChatCompletionClientBase`

- **get_chat_message_contents** (non-streaming):  
  - Auto-invokes function calls (FCC), appends results (FRC).
  - Returns final message list, depending upon number of response choices (usually one response only).
  - FCC/FRC contained inside of ChatHistory, unless a filter is configured then it's FRC returned.
- **get_streaming_chat_message_contents** (streaming):  
  - Streams chunks (text + FCC).
  - Auto-invokes FCC mid-stream; FRC may show up in stream, if filter configured.
  - No `is_visible` as some agent code has; emits all content as-is.

---

#### Agent Layer Behavior Differences

##### ChatCompletionAgent

- `get_response()`: Final message only (FCC/FRC included if filter present).
- `invoke()`: Yields each ChatMessageContent (FCC/FRC contained in ChatHistory, accessible via thread.get_messages()).
- `invoke_stream()`: Streams chunks; no filtering, (text + FCC content visible).
- No `is_visible` concept—everything is included.

##### AzureAIAgent / OpenAIAssistantAgent

- Use `is_visible=False` for internal steps (FCC, code execution).
- Yielded content format:
  - `(False, ChatMessageContent(...FCC...))`
  - `(True, ChatMessageContent(...final text...))`
- Function calls are hidden unless explicitly exposed.

##### BedrockAgent

- Similar to Azure/OpenAI:
  - Function calls may not be yielded at all.
  - Yields only final user-facing text with `is_visible=True`.

##### AutoGenConversableAgent

- No streaming or `is_visible`.
- `get_response()`: Single final message.
- `invoke()`: Yields multiple steps; no partial streaming or visibility control.

---

#### Summary of Differences

- `is_visible` usage:
  - Azure, OpenAI, Bedrock: selectively hide FCC/FRC.
  - ChatCompletionAgent, AutoGen: expose all content directly.
- Streaming support:
  - Only ChatCompletionAgent streams all content without filtering.
- FCC/FRC exposure:
  - Varies: some agents yield them directly, others hide or merge them.
  - For AzureAIAgent / OpenAIAssistantAgent - thread.get_messages() only gets simple text, does not include FCC

---

#### Proposal for server-side agents (AzureAIAgent / OpenAIAssistantAgent)

- Include `on_complete` callback similar to `invoke_stream` where if provided,
  we return the various content types to the user.
- Question is if we want to handle it this way only for "server-side" agents
  or if we'd do the same for the ChatCompletionAgent.
