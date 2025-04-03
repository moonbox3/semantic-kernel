# Multi-agent orchestration feature requirements

1. Define orchestration pattern input and output
      - Sync (return when the pattern finishes) vs Async (return immediately)
        - Considerations: easy to get started vs scalability
        - Sync: return the result of the pattern when it finishes
        - Async: broadcast the result of the pattern when it finishes
      - Arbitrary structured (user-defined) input and output
        - Considerations: Being able to nest patterns, i.e. the output of one pattern is the input of another
        - Input: a list of tasks with additional attributes for context
        - Output: the result of the pattern, which can be a list of tasks or a single task, and a context object
        - We probably cannot use the ChatMessageContent type for the following reasons
          - The name implies chat messages, which is no longer applicable for multi-agent orchestration
          - Some properties are not applicable for multi-agent orchestration, such as role and content
2. Nested patterns
      - Considerations: being able to nest patterns that can create very complex workflows
      - Patterns need to have the same invocation signature
      - Patterns need to accept SK agents and arbitrary patterns at construction time
      - Runtime
          - Patterns shouldn't not be aware of the runtime. The runtime should be provided when the pattern is invoked.
          - Register the agents and patterns in the runtime before the execution starts.
          - Add subscriptions to the runtime before the execution starts.
          - Remove registrations and subscriptions from the runtime after the execution finishes to avoid name collisions.
3. User Proxy
      - Considerations: keep the user in the loop and allow them to intervene in the orchestration process
4. Save states and rehydration
      - Considerations: being able to save the state of the orchestration process while waiting for user input and restore it later when user provides input for scalability
      - Recursively save the state of all agents and child patterns, including threads, chat history, and context from the root pattern
