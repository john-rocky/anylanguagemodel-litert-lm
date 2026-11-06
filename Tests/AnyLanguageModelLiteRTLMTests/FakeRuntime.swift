import AnyLanguageModel
import Foundation

@testable import AnyLanguageModelLiteRTLM

/// What a fake conversation does with the message it gets.
enum FakeBehavior: Sendable {
  /// Replies with this text (streamed as one delta per element of `deltas` when set).
  case reply(String)
  /// Streams these deltas, or replies with them joined.
  case deltas([String])
  /// Throws this error.
  case fail(FakeNativeError)
  /// Streams these deltas, then waits until `cancel()` and fails like LiteRT-LM does.
  case waitForCancel(deltasBefore: [String])
}

/// An error with LiteRT-LM's description, as the mapping reads it.
struct FakeNativeError: LocalizedError, Sendable {
  let message: String
  var errorDescription: String? { message }
}

/// The record of one message sent to a fake conversation.
struct SentMessage: Equatable, Sendable {
  var text: String
  var maxOutputTokens: Int?
  var schemaJSON: String?
  var streamed: Bool
}

/// A lock-protected value for the fakes.
final class Guarded<Value>: @unchecked Sendable {
  private var value: Value
  private let lock = NSLock()

  init(_ value: Value) { self.value = value }

  func withLock<R>(_ body: (inout Value) throws -> R) rethrows -> R {
    lock.lock()
    defer { lock.unlock() }
    return try body(&value)
  }

  var current: Value { withLock { $0 } }
}

/// Fires once; `wait()` returns after it fired.
final class Signal: @unchecked Sendable {
  private var fired = false
  private var waiters: [CheckedContinuation<Void, Never>] = []
  private let lock = NSLock()

  func fire() {
    lock.lock()
    fired = true
    let waiting = waiters
    waiters = []
    lock.unlock()
    for waiter in waiting { waiter.resume() }
  }

  func wait() async {
    await withCheckedContinuation { continuation in
      lock.lock()
      if fired {
        lock.unlock()
        continuation.resume()
      } else {
        waiters.append(continuation)
        lock.unlock()
      }
    }
  }
}

final class FakeConversation: RuntimeConversation, @unchecked Sendable {
  let behavior: FakeBehavior
  let benchmarkValue: RuntimeBenchmark?
  let sent = Guarded<[SentMessage]>([])
  let cancelCount = Guarded(0)
  let cancelled = Signal()
  /// Fires when the message arrives.
  let started = Signal()
  private let inFlight: Guarded<Int>
  private let maxInFlight: Guarded<Int>

  init(
    behavior: FakeBehavior, benchmark: RuntimeBenchmark?, inFlight: Guarded<Int>,
    maxInFlight: Guarded<Int>
  ) {
    self.behavior = behavior
    self.benchmarkValue = benchmark
    self.inFlight = inFlight
    self.maxInFlight = maxInFlight
  }

  private func begin(_ message: SentMessage) {
    sent.withLock { $0.append(message) }
    let now = inFlight.withLock { count -> Int in
      count += 1
      return count
    }
    maxInFlight.withLock { $0 = max($0, now) }
    started.fire()
  }

  private func end() {
    inFlight.withLock { $0 -= 1 }
  }

  func send(_ text: String, maxOutputTokens: Int?, jsonSchema: String?) async throws -> String {
    begin(SentMessage(text: text, maxOutputTokens: maxOutputTokens, schemaJSON: jsonSchema, streamed: false))
    defer { end() }
    switch behavior {
    case .reply(let reply):
      return reply
    case .deltas(let deltas):
      return deltas.joined()
    case .fail(let error):
      throw error
    case .waitForCancel:
      await cancelled.wait()
      throw FakeNativeError(message: "CANCELLED: Task cancelled")
    }
  }

  func stream(_ text: String, maxOutputTokens: Int?, jsonSchema: String?)
    -> AsyncThrowingStream<String, any Error>
  {
    begin(SentMessage(text: text, maxOutputTokens: maxOutputTokens, schemaJSON: jsonSchema, streamed: true))
    let behavior = self.behavior
    let cancelled = self.cancelled
    return AsyncThrowingStream { continuation in
      Task {
        defer { self.end() }
        switch behavior {
        case .reply(let reply):
          continuation.yield(reply)
          continuation.finish()
        case .deltas(let deltas):
          for delta in deltas { continuation.yield(delta) }
          continuation.finish()
        case .fail(let error):
          continuation.finish(throwing: error)
        case .waitForCancel(let deltasBefore):
          for delta in deltasBefore { continuation.yield(delta) }
          await cancelled.wait()
          continuation.finish(throwing: FakeNativeError(message: "CANCELLED: Task cancelled"))
        }
      }
    }
  }

  func cancel() {
    cancelCount.withLock { $0 += 1 }
    cancelled.fire()
  }

  func benchmark() -> RuntimeBenchmark? { benchmarkValue }
}

final class FakeEngine: LoadedEngine, @unchecked Sendable {
  /// The behavior of each new conversation, in order; the last one repeats.
  let behaviors: Guarded<[FakeBehavior]>
  let benchmark: RuntimeBenchmark?
  let setups = Guarded<[ConversationSetup]>([])
  let conversations = Guarded<[FakeConversation]>([])
  let inFlight = Guarded(0)
  let maxInFlight = Guarded(0)

  init(_ behaviors: [FakeBehavior], benchmark: RuntimeBenchmark? = nil) {
    self.behaviors = Guarded(behaviors)
    self.benchmark = benchmark
  }

  func makeConversation(_ setup: ConversationSetup) async throws -> any RuntimeConversation {
    setups.withLock { $0.append(setup) }
    let behavior = behaviors.withLock { list -> FakeBehavior in
      list.count > 1 ? list.removeFirst() : list[0]
    }
    let conversation = FakeConversation(
      behavior: behavior, benchmark: benchmark, inFlight: inFlight, maxInFlight: maxInFlight)
    conversations.withLock { $0.append(conversation) }
    return conversation
  }

  var lastSetup: ConversationSetup? { setups.current.last }
  var lastSent: SentMessage? { conversations.current.last?.sent.current.last }
}

final class FakeLoader: EngineLoading, @unchecked Sendable {
  /// Errors for the next loads, in order; after them a load succeeds.
  let failures: Guarded<[FakeNativeError]>
  let engine: FakeEngine
  let loadCount = Guarded(0)
  let specs = Guarded<[EngineSpec]>([])
  /// How long a load takes, to overlap loads in a test.
  let delayNanoseconds: UInt64

  init(engine: FakeEngine, failures: [FakeNativeError] = [], delayNanoseconds: UInt64 = 0) {
    self.engine = engine
    self.failures = Guarded(failures)
    self.delayNanoseconds = delayNanoseconds
  }

  func loadEngine(_ spec: EngineSpec) async throws -> any LoadedEngine {
    loadCount.withLock { $0 += 1 }
    specs.withLock { $0.append(spec) }
    if delayNanoseconds > 0 { try? await Task.sleep(nanoseconds: delayNanoseconds) }
    if let failure = failures.withLock({ $0.isEmpty ? nil : $0.removeFirst() }) {
      throw failure
    }
    return engine
  }
}

/// A model over a fake runtime, with a real (empty) model file so the file check passes.
struct FakeSetup {
  let model: LiteRTLMLanguageModel
  let loader: FakeLoader
  let store: EngineStore
  var engine: FakeEngine { loader.engine }

  init(
    _ behaviors: [FakeBehavior] = [.reply("Hello")],
    failures: [FakeNativeError] = [],
    benchmark: RuntimeBenchmark? = nil,
    loadDelayNanoseconds: UInt64 = 0
  ) throws {
    let directory = FileManager.default.temporaryDirectory
      .appendingPathComponent("alm-litertlm-tests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let file = directory.appendingPathComponent("fake.litertlm")
    try Data("not a model".utf8).write(to: file)
    loader = FakeLoader(
      engine: FakeEngine(behaviors, benchmark: benchmark), failures: failures,
      delayNanoseconds: loadDelayNanoseconds)
    store = EngineStore(loader: loader)
    model = LiteRTLMLanguageModel(
      source: .file(file), backend: .gpu, maxTokens: 2048, cacheDirectory: directory,
      enableBenchmark: false, onDownloadProgress: nil, store: store)
  }
}

@Generable(description: "A short review of a cafe")
struct CafeReview {
  @Guide(description: "The cafe's name")
  var name: String
  @Guide(description: "Rating from 1 to 5", .range(1...5))
  var rating: Int
  @Guide(description: "Price level", .anyOf(["cheap", "moderate", "expensive"]))
  var price: String
  @Guide(description: "Whether the cafe has free wifi")
  var hasWifi: Bool
}

@Generable(description: "A stop on a trip")
struct Stop {
  @Guide(description: "Name of the place")
  var place: String
  @Guide(description: "Minutes to spend there", .range(15...240))
  var minutes: Int
  @Guide(description: "Kind of place", .anyOf(["temple", "museum", "park", "restaurant", "shop"]))
  var kind: String
}

@Generable(description: "A one-day trip plan")
struct TripPlan {
  @Guide(description: "Title of the plan")
  var title: String
  @Guide(description: "The first stop of the day")
  var start: Stop
  @Guide(description: "The other stops, in order", .count(1...4))
  var stops: [Stop]
  @Guide(description: "Whether the plan needs a car")
  var needsCar: Bool
}

let cafeJSON = #"{"hasWifi":true,"name":"The Daily Grind","price":"moderate","rating":4}"#

/// The schema string the backend sends for `CafeReview`, for the native probe.
func cafeReviewSchemaJSON() throws -> String {
  try StructuredOutput.schemaJSON(for: CafeReview.generationSchema)
}
