import Foundation
import LiteRTLM

// The two seams between the backend and the native runtime: a loaded engine and one
// request's conversation. The unit tests put fakes here; the app gets the LiteRTLM types.

/// What identifies one loaded engine. Two models with the same spec share an engine.
struct EngineSpec: Hashable, Sendable {
  var modelPath: String
  var backend: Backend
  var maxTokens: Int
  var cacheDirectory: String?
  var enableBenchmark: Bool
}

/// A message before the current prompt.
struct HistoryMessage: Equatable, Sendable {
  enum Role: Equatable, Sendable {
    case user
    case model
  }

  var role: Role
  var text: String
}

/// The sampler of one request, in the units LiteRT-LM takes.
struct SamplerSettings: Equatable, Sendable {
  var topK: Int
  var topP: Float
  var temperature: Float
  var seed: Int
}

/// Everything a new conversation starts with.
struct ConversationSetup: Equatable, Sendable {
  var systemMessage: String?
  var history: [HistoryMessage]
  var sampler: SamplerSettings
  /// True when the request sends a JSON schema (LiteRT-LM's constrained decoding).
  var constrained: Bool
}

/// The engine's numbers for the last turn of a conversation.
struct RuntimeBenchmark: Equatable, Sendable {
  var prefillTokens: Int
  var decodeTokens: Int
  var prefillTokensPerSecond: Double
  var decodeTokensPerSecond: Double
  var timeToFirstTokenSeconds: Double
  /// Tokens in the conversation's KV cache after the turn (prompt and reply), if known.
  var kvCacheTokens: Int?
}

protocol EngineLoading: Sendable {
  func loadEngine(_ spec: EngineSpec) async throws -> any LoadedEngine
}

protocol LoadedEngine: AnyObject, Sendable {
  func makeConversation(_ setup: ConversationSetup) async throws -> any RuntimeConversation
}

protocol RuntimeConversation: AnyObject, Sendable {
  /// Generates the whole reply. Returns after the native generation ended.
  func send(_ text: String, maxOutputTokens: Int?, jsonSchema: String?) async throws -> String
  /// Streams text deltas. The stream finishes after the native generation ended.
  func stream(_ text: String, maxOutputTokens: Int?, jsonSchema: String?)
    -> AsyncThrowingStream<String, any Error>
  /// Asks the native runtime to stop the generation in flight.
  func cancel()
  /// The engine's numbers for the last turn, when the engine was loaded with benchmarking on.
  func benchmark() -> RuntimeBenchmark?
}

// MARK: - LiteRTLM

struct LiteRTLMEngineLoader: EngineLoading {
  func loadEngine(_ spec: EngineSpec) async throws -> any LoadedEngine {
    if spec.enableBenchmark {
      // A process-wide flag that the engine reads when it is created.
      ExperimentalFlags.optIntoExperimentalAPIs()
      ExperimentalFlags.enableBenchmark = true
    }
    let config = try EngineConfig(
      modelPath: spec.modelPath,
      backend: spec.backend,
      maxNumTokens: spec.maxTokens,
      cacheDir: spec.cacheDirectory)
    let engine = Engine(engineConfig: config)
    try await engine.initialize()
    return LiteRTLMEngine(engine: engine, benchmark: spec.enableBenchmark)
  }
}

final class LiteRTLMEngine: LoadedEngine {
  private let engine: Engine
  private let benchmark: Bool

  init(engine: Engine, benchmark: Bool) {
    self.engine = engine
    self.benchmark = benchmark
  }

  func makeConversation(_ setup: ConversationSetup) async throws -> any RuntimeConversation {
    let sampler = try SamplerConfig(
      topK: setup.sampler.topK,
      topP: setup.sampler.topP,
      temperature: setup.sampler.temperature,
      seed: setup.sampler.seed)
    let config = ConversationConfig(
      systemMessage: setup.systemMessage.map { Message($0, role: .system) },
      initialMessages: setup.history.map { Message($0.text, role: $0.role == .user ? .user : .model) },
      samplerConfig: sampler,
      automaticToolCalling: false,
      enableResponseFormat: setup.constrained)
    let conversation = try await engine.createConversation(with: config)
    return LiteRTLMConversation(conversation: conversation, benchmark: benchmark)
  }
}

final class LiteRTLMConversation: RuntimeConversation {
  private let conversation: Conversation
  private let benchmarkEnabled: Bool

  init(conversation: Conversation, benchmark: Bool) {
    self.conversation = conversation
    self.benchmarkEnabled = benchmark
  }

  func send(_ text: String, maxOutputTokens: Int?, jsonSchema: String?) async throws -> String {
    let format = try jsonSchema.map { try ResponseFormat.json(schema: $0) }
    let reply = try await conversation.sendMessage(
      Message(text), maxOutputTokens: maxOutputTokens, responseFormat: format)
    return reply.toString
  }

  func stream(_ text: String, maxOutputTokens: Int?, jsonSchema: String?)
    -> AsyncThrowingStream<String, any Error>
  {
    let conversation = self.conversation
    return AsyncThrowingStream { continuation in
      // The wrapper's stream does not stop the native generation when its reader goes away,
      // so this task reads it to the end, and a reader that leaves cancels the generation.
      Task {
        do {
          let format = try jsonSchema.map { try ResponseFormat.json(schema: $0) }
          let chunks = conversation.sendMessageStream(
            Message(text), maxOutputTokens: maxOutputTokens, responseFormat: format)
          for try await chunk in chunks {
            let delta = chunk.toString
            if !delta.isEmpty { continuation.yield(delta) }
          }
          continuation.finish()
        } catch {
          continuation.finish(throwing: error)
        }
        // Keep the conversation alive until the stream ended, so that its last reference does
        // not fall on the runtime's callback thread.
        withExtendedLifetime(conversation) {}
      }
      continuation.onTermination = { termination in
        if case .cancelled = termination { try? conversation.cancel() }
      }
    }
  }

  func cancel() {
    try? conversation.cancel()
  }

  func benchmark() -> RuntimeBenchmark? {
    guard benchmarkEnabled, ExperimentalFlags.enableBenchmark,
      let info = try? conversation.getBenchmarkInfo()
    else { return nil }
    return RuntimeBenchmark(
      prefillTokens: info.lastPrefillTokenCount,
      decodeTokens: info.lastDecodeTokenCount,
      prefillTokensPerSecond: info.lastPrefillTokensPerSecond,
      decodeTokensPerSecond: info.lastDecodeTokensPerSecond,
      timeToFirstTokenSeconds: info.timeToFirstTokenInSecond,
      kvCacheTokens: try? conversation.getTokenCount())
  }
}
