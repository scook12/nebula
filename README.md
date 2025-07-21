# Nebula

**Intelligent system services powered by AI/ML hardware acceleration**

Nebula is a framework for building AI-powered system agents that optimize computing resources through machine learning. It provides hardware abstraction for NPUs (Neural Processing Units), a unified Agent SDK, and intelligent agents for filesystem optimization, resource management, and system automation.

Built for RedoxOS <3

## 🌟 What is Nebula?

Nebula is an **AI agent framework** designed to provide a _runtime for intelligent system services_.

### **On Microkernel Systems (like RedoxOS):**
- Collection of intelligent userspace services and schemes  
- Leverages microkernel isolation for secure AI processing
- Implements system optimization through ML-powered agents
- Provides developers a way to approach Agent chaos with ease

### **On Traditional Systems (macOS, Linux, Windows, supported as a best effort):**
- AI agent runtime with NPU hardware acceleration
- Intelligent system monitoring and optimization layer
- ML-powered automation and prediction services

### **Core Value Proposition:**
- **Hardware Abstraction**: Unified interface for AI/ML accelerators (Apple Neural Engine, Intel NPU, NVIDIA GPU, etc.)
- **Agent Framework**: SDK for building intelligent system services
- **ML Integration**: Easy ONNX model deployment and inference
- **System Intelligence**: Predictive optimization of filesystem, memory, and compute resources

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Nebula Agent Framework                   │
├─────────────────────────────────────────────────────────────┤
│  Intelligent Agents  │    Agent SDK     │  Hardware Support │
│  ┌─────────────────┐  │ ┌─────────────┐  │ ┌───────────────┐ │
│  │ Filesystem      │  │ │ ML Runtime  │  │ │ Apple Neural  │ │
│  │ Agent           │  │ │ IPC/Comms   │  │ │ Engine        │ │
│  ├─────────────────┤  │ │ Scheduling  │  │ ├───────────────┤ │
│  │ Memory Agent    │  │ │ Monitoring  │  │ │ Intel NPU     │ │
│  ├─────────────────┤  │ └─────────────┘  │ ├───────────────┤ │
│  │ Network Agent   │  │                  │ │ NVIDIA GPU    │ │
│  └─────────────────┘  │                  │ └───────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                   Host Operating System                     │
│              (Redox! [macOS, Linux, Windows)                │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
```bash
# Rust toolchain (latest stable)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Clone Nebula
git clone <repository-url>
cd nebula
```

### Try the NPU Hardware Abstraction Layer
```bash
cd agent_sdk

# Test NPU detection and capabilities
cargo run --example npu_demo --features npu

# Test Apple Neural Engine (macOS only)
cargo run --example apple_neural_engine_demo --features "npu,apple_neural_engine"
```

### Run the Filesystem Agent
```bash
cd agents/filesystem_agent

# Build and run with ML-based file access prediction
cargo run --features ai
```

### Explore the Agent SDK
```bash
cd agent_sdk

# Run tests
cargo test --features "ai,npu"

# Browse examples
ls examples/
```

## 📦 Components

### 🧠 [Agent SDK](agent_sdk/)
Unified framework for building intelligent system agents:
- **ML Runtime**: ONNX model loading and inference
- **NPU HAL**: Hardware abstraction for AI accelerators  
- **IPC Framework**: Inter-agent communication
- **Async Runtime**: Tokio-based agent execution environment

### 🤖 [Intelligent Agents](agents/)
Production-ready agents that optimize system resources:
- **[Filesystem Agent](agents/filesystem_agent/)**: ML-based file access prediction and caching
- **[Agent Daemon](agents/agentd/)**: Core agent management and IPC coordination

### 🔧 [ML Models](ml_models/)
Machine learning models for system optimization:
- File access pattern prediction (ONNX)
- Resource utilization forecasting
- Anomaly detection for system health

## 🎯 Use Cases

### **System Administrators**
- Predictive resource scaling
- Intelligent log analysis and alerting
- Automated performance optimization

### **Developers**
- Build AI-powered system tools
- Integrate ML models with system services
- Leverage NPU hardware for acceleration

### **Researchers**
- Experiment with AI-driven system optimization
- Prototype intelligent resource management
- Study machine learning in systems contexts

## 💻 Platform Support

| Platform | NPU Support | Agent Runtime | Status |
|----------|-------------|---------------|---------|
| **macOS (Apple Silicon)** | ✅ Apple Neural Engine | ✅ Full | Ready |
| **macOS (Intel)** | 🔶 CPU Fallback | ✅ Full | Ready |
| **Linux** | 🚧 Intel NPU (planned) | ✅ Full | Ready |
| **Windows** | 🚧 Intel NPU (planned) | 🔶 Limited | Development |
| **Redox OS** | ✅ Mock + CPU | 🚧 Integration | Development |

## 📚 Documentation

- **[NPU Hardware Abstraction Layer](agent_sdk/NPU_HAL_README.md)** - Complete guide to NPU integration
- **[Development Setup](DEVELOPMENT_SETUP.md)** - Environment configuration and workflow
- **[Milestones](milestones/)** - Development history and roadmap
- **[Agent Examples](agent_sdk/examples/)** - Usage examples and tutorials

## 🛣️ Roadmap and Status

We're at the end of our proof-of-concept effort. We've created an Agent SDK, a Hardware Abstraction Layer (HAL), and an ONNX-based model runtime with a simple filesystem optimization agent. This project is not production ready and requires a lot more effort and testing on the hardware side. 
Most of our success and demonstrations prove the architecture functions. You can build intelligent system services on a microkernel in user-space that make use of an abstract layer for hardware acceleration + provide a clean and secure interface for application-tier consumption of agents via IPC.
With that said, most of the code at this stagerelies on mocked implementations of the underlying systems and half-implemented consumers. There's a long way to go!

### Current (Phase 1) ✅
- [x] Agent SDK foundation
- [x] NPU Hardware Abstraction Layer
- [x] Apple Neural Engine mock support
- [x] Filesystem optimization agent
- [x] ONNX model integration

### Near Term (Phase 2) 🚧
- [ ] Redox OS integration
- [ ] Extended NPU support
- [ ] Advanced memory management agent
- [ ] Network optimization agent
- [ ] Performance benchmarking suite

### Future (Phase 3) 📋
- [ ] Multi-device inference coordination
- [ ] Distributed agent orchestration
- [ ] Cloud NPU federation
- [ ] Real-time system adaptation
- [ ] Advanced security and isolation

## 🤝 Contributing

We welcome contributions! Please see our development workflow:

1. **Start with the Agent SDK**: Understand the core framework
2. **Explore existing agents**: Learn from filesystem and daemon implementations
3. **Try NPU integration**: Test hardware abstraction on your platform
4. **Build something cool**: Create new agents or enhance existing ones

## 📄 License

Licensed under MIT License. See [LICENSE](LICENSE) for details.

## 🌟 Why Nebula?

Agents have taken the world by storm and are automating workflows at a rapid pace. But there's a lot of architectural issues with the way that agents currently work and we've only just begun to see the extent that they can be used to improve our day-to-day lives.

Traditional systems are reactive - they respond to resource demands after problems occur. **Nebula makes systems proactive** by using machine learning to predict, optimize, and adapt system behavior before issues arise. This is the value proposition of intelligent system services and having a secure way to do this is crucial for the next wave of AI adoption.

By leveraging specialized AI hardware (NPUs), Nebula can perform complex optimizations with minimal impact on system resources, creating genuinely intelligent computing environments.

Moreover, as developers begin to push their agents further into a user's system, they need wider and broader access. Ever seen the list of permissions an AI Agent wants for your Google account?
MCP Servers are one approach to solving this problem, but they have issues with performance and security. They also provide a protocol for external applications, but only to adapt your runtime to a model's interface. This approach also mostly benefits the Agent provider -- you build the integration to put the Agent interface in front of yours. Doing things the other way around is currently _impossible_ to do cleanly and securely.

Nebula provides a unified, accelerated runtime for agents _and_ a secure framework for integrating with those agents on RedoxOS via system services in the user space instead.

---

*Built with ❤️ for the future of intelligent computing*
