---
# These are optional elements. Feel free to remove any of them.
status: proposed
contact: evmattso
date: 2025-04-10
deciders: bentho, markwallace, sphenry
---

# Integrating A2A Protocol with Semantic Kernel

## 1. Executive Summary

This report examines the Agent-to-Agent (A2A) protocol, a standardized JSON-RPC based framework for orchestrating multi-turn, real-time, and stateful interactions between autonomous agents, and explores how Microsoft’s Semantic Kernel (SK) can align with and extend A2A capabilities. It outlines three primary integration pathways:

- Implementing the A2A protocol within the SK framework.
- Enabling SK Agents to operate as client agents that initiate A2A requests.
- Configuring an SK Agent to serve as an A2A server agent, managing incoming agent communications.

By leveraging SK’s modular function orchestration, current agent/process framework, dynamic memory management, and integration with external tools, an organization can build robust, conversational AI systems that participate in complex agent collaborations.

## 2. Overview of the A2A Protocol

### 2.1. Core Features

The A2A protocol is engineered to manage rich, interactive communications among autonomous agents. Its key features include:

- **Multi-turn Conversations**: The protocol supports dialog flows where agents can clarify user intents, request additional data, or update context mid-conversation. For instance, a currency conversion query may trigger a clarifying exchange before providing a final answer.

- **Real-time Streaming**: Agents benefit from incremental status updates during processing. Rather than waiting for a complete response, consumers receive streaming feedback (e.g., “Looking up rates…” then “Processing…”), enhancing transparency.

- **Push Notifications**: Leveraging webhook-based push systems, the protocol allows asynchronous communication. These notifications can alert client systems or collaborating agents upon key events (e.g., task completion or error conditions).

- **Conversational Memory**: A session-based memory mechanism preserves context over multiple interactions. This is especially important in dynamic conversations where decisions depend on prior exchanges.

- **External Tool Integration**: The protocol can seamlessly integrate specialized tools (such as a currency conversion service via the Frankfurter API), demonstrating its extensibility beyond simple text-based tasks.

### 2.2. Agent Discovery and Agent Cards

A2A includes a flexible mechanism for discovering agents via standardized **AgentCards** -- JSON documents describing an agent’s capabilities, endpoints, and identity. These cards serve as the foundation for dynamic agent discovery and are critical for enabling agent-to-agent collaboration across organizational and platform boundaries.

**Agent Discovery Methods:**

- **Open Discovery**: Enterprises can expose an AgentCard at a well-known path (`https://DOMAIN/.well-known/agent.json`), allowing clients to retrieve agent metadata via simple HTTP GET requests. This enables discovery using DNS and fosters an open, decentralized discovery mechanism.

- **Curated Discovery (Registry-Based)**: Organizations may maintain internal registries or catalogs of trusted agents. This curated model supports governance, access control, and enterprise-specific configurations while promoting agent reuse across departments.

- **Private Discovery (API-Based)**: Proprietary agents may require specialized API interfaces for discovery. These methods lie outside the core A2A specification but are anticipated in secure, private deployments.

**Security Considerations:**

AgentCards can contain sensitive information, including credentials or internal endpoints. Implementors should secure access using mechanisms like mTLS, OAuth, or identity-aware proxies. Even well-known paths may require authentication to protect organizational boundaries.

Related Links:

- [A2A Protocol](https://google.github.io/A2A/#/)
- [A2A Python Code Samples](https://github.com/google/A2A/tree/main/samples/python)

## 3. Overview of Microsoft’s SK

SK is an extensible, model-agnostic SDK that enables the orchestration of LLM models, AI agents, and external native-code plugins. Its distinguishing features include:

- **Modular Plugin Integration**: Through a plugin architecture, SK allows the integration of both native and semantic functions (referred to as “skills” in A2A) -- these can encapsulate external APIs, custom logic, or dynamic prompts.

- **Dynamic Memory Management**: SK supports transient and persistent memory, enabling agents to maintain context across interactions -- a critical capability aligning with A2A’s conversational memory.

- **Adaptive Function Orchestration**: SK’s design facilitates function discovery, planning, and dynamic selection. This enables the AI to decide, based on context, which plugins to invoke—similar in spirit to A2A’s dynamic tool discovery.

- **Model Agnosticism and Extensibility**: SK can interface with various LLMs, whether from OpenAI, Azure, or custom deployments. This makes it suitable as an integration layer within heterogeneous AI ecosystems.

- **AgentThread for Stateful Dialog Management**: SK's `AgentThread` abstraction provides a structured container for managing ongoing, multi-turn conversations. It encapsulates not only message history but also the execution state of a conversation thread. This aligns directly with A2A’s emphasis on thread and session-based conversation management, enabling consistent, contextual exchanges between agents over time. By mapping A2A sessions to `AgentThread` instances, developers can simplify the management of long-running dialog flows and preserve agent memory within defined thread scopes.

## 4. Integration Pathways: How SK Can Fit with A2A

To align with the A2A protocol, SK can be tailored and extended in several ways. Each pathway addresses specific architectural roles:

### 4.1. Implementing the A2A Protocol within SK

Create an A2A connector within the SK SDK that directly processes JSON-RPC A2A messages.

**Approach**

- **Adapter Layer**: Develop a connector that parses incoming A2A-compliant JSON-RPC requests. This layer maps message components (session IDs, roles, accepted output modes, and message parts) to SK related content types.
- **Stateful Management**: Utilize SK’s thread and memory abstractions to persist session data, thereby replicating the A2A protocol’s checkpoint memory.
- **Streaming and Notification**: Incorporate asynchronous handlers in SK to emit periodic "working" state messages, and integrate webhook modules to dispatch push notifications upon event completions.

**Benefits**

- A unified framework for managing both human-like language interactions and external tool invocations.
- The ability to standardize agent interactions internally while adhering to a broader ecosystem standard.

### 4.2. SK Agents Acting as A2A Client Agents

Enable existing SK Agents to issue A2A-style client requests to external agents or services that also adopt the A2A protocol.

**Technical Approach**

- **Client Library Extension**: Augment SK’s agent interface with a client module that supports A2A-compliant request generation. This module would package an SK Agent’s commands into JSON-RPC messages as defined by A2A.
- **Multi-turn Interaction**:  Use SK’s built-in support for tool invocation and conversational state to handle structured back-and-forth exchanges. This allows agents to pause for additional input, retry failed operations, or clarify ambiguous prompts -- mirroring A2A’s expectation that agents negotiate task completion through tool-driven, stateful dialogue.
- **Feedback Integration**: Integrate support for handling intermediate status updates or structured incremental responses from external agents via the A2A JSON-RPC protocol. This allows SK agents to react dynamically to ongoing processing states from their counterparts, enhancing transparency and responsiveness beyond just model-based streaming.

**Benefits**

- SK Agents can seamlessly interface with external autonomous agents, fostering collaboration across different platforms.
- The client agent role ensures that existing SK capabilities remain agnostic to the underlying A2A communication standards.

### 4.3. SK Agent as an A2A Server Agent
 
Configure an SK agent to serve as an A2A server, exposing its rich functionality via the A2A protocol.

**Technical Approach**

- **Server Endpoint Implementation**: Develop server endpoints within an SK connector that accept JSON-RPC formatted calls from other agents. This endpoint would translate incoming calls into SK function invocations.
- **Dynamic Skill Exposure**: Use SK’s plugin architecture to publish available A2A skills (or “tools”) along with descriptive metadata, allowing remote agents to dynamically discover and invoke these functions.
- **Session and Context Management**: Integrate SK’s thread and memory abstractions to track session-specific context and ensure that multi-turn dialogues initiated by external agents are handled appropriately.

**Benefits**

- This approach transforms SK into a service provider, broadening its applicability in enterprise ecosystems that rely on standardized agent communications.
- A server SK Agent can participate in orchestrated multi-agent systems, enabling hybrid workflows where certain agents serve as specialized function providers.
- The SK server agent can also act as a delegator, intelligently breaking down incoming tasks into smaller subtasks and assigning these to specialized client agents for execution, thereby facilitating scalable, distributed task management.

## 5. Strategic Considerations and Business Impact

### 5.1. Enhanced Interoperability

By aligning SK with the A2A protocol, organizations gain a standardized interface for agent communication. This promotes interoperability not only among proprietary systems but also with third-party AI agents that adhere to the same protocol, ultimately reducing integration friction.

### 5.2. Flexibility and Modularity

SK’s modular design allows it to seamlessly function either as an A2A client, initiating requests, or as an A2A server, handling incoming agent interactions. This flexibility enables organizations to quickly adapt and reconfigure their agent workflows in response to evolving business requirements or changing operational contexts.

### 5.3. Real-time Responsiveness and Transparency

With built-in support for streaming responses and push notifications, integrating A2A into SK-enhanced systems delivers a more engaging and transparent user experience. Decision-makers benefit from real-time status updates and can rapidly identify and resolve bottlenecks.

### 5.4. Future-Proof Architecture

Adopting standards like A2A positions an enterprise to leverage emerging trends in autonomous agent communication. The alignment of SK with A2A ensures that the AI infrastructure remains flexible and adaptive, supporting advanced multi-agent collaborations and continuous innovation.

## 6. Implementation Roadmap

It would take 2–3 weeks to properly create the SK Client and Server Agents following the A2A protocol. Most of the time would be spent testing various scenarios that involve network discovery for clients and having the SK Agent act as a host. Additional work around exploring various authentication scenarios is also needed.

## 7. Conclusion

Integrating the A2A protocol with SK presents a strategic opportunity to standardize and streamline inter-agent communications in a stateful, dynamic manner. Whether by implementing an A2A adapter within SK, empowering SK Agents to operate as client agents, or configuring an SK Agent as an A2A server, this alignment enhances interoperability, real-time responsiveness, and strategic flexibility. Such an integration will not only position the organization at the forefront of conversational AI innovation but also facilitate a more agile and resilient multi-agent ecosystem for achieving business objectives.
