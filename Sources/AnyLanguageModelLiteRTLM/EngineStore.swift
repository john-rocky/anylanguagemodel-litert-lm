import Foundation
import os

/// Holds one engine per `EngineSpec` for the whole process.
///
/// A load is single-flight: requests that arrive while a load runs wait for the same load.
/// A failed load is remembered for `availability` and tried again by the next request.
actor EngineStore {
  static let shared = EngineStore(loader: LiteRTLMEngineLoader())

  /// Failure messages that `availability` reads synchronously.
  nonisolated let failures = LoadFailures()

  private let loader: any EngineLoading
  private var states: [EngineSpec: State] = [:]
  private var nextLoadID: UInt64 = 0

  private enum State {
    case loading(id: UInt64, task: Task<EngineEntry, any Error>)
    case ready(EngineEntry)
  }

  init(loader: any EngineLoading) {
    self.loader = loader
  }

  /// The engine for `spec`, loading it first if needed. `fetch` runs inside the load,
  /// before the engine is created (the download of a catalog or Hugging Face model).
  func entry(
    for spec: EngineSpec,
    fetch: (@Sendable () async throws -> Void)? = nil
  ) async throws -> EngineEntry {
    switch states[spec] {
    case .ready(let entry):
      return entry
    case .loading(_, let task):
      return try await task.value
    case nil:
      break
    }
    nextLoadID &+= 1
    let id = nextLoadID
    let loader = self.loader
    let task = Task<EngineEntry, any Error> {
      try await fetch?()
      return EngineEntry(engine: try await loader.loadEngine(spec))
    }
    states[spec] = .loading(id: id, task: task)
    do {
      let entry = try await task.value
      if case .loading(let current, _) = states[spec], current == id {
        states[spec] = .ready(entry)
        failures.clear(spec)
      }
      return entry
    } catch {
      if case .loading(let current, _) = states[spec], current == id {
        states[spec] = nil
        failures.record(ErrorMapping.describe(error), for: spec)
      }
      throw error
    }
  }

  /// Drops the store's engine for `spec` and forgets a failed load. Requests in flight keep
  /// their engine until they end; the native memory is freed after the last one.
  func unload(_ spec: EngineSpec) {
    states[spec] = nil
    failures.clear(spec)
  }

  /// Whether the store holds a loaded engine for `spec` (for tests).
  func isLoaded(_ spec: EngineSpec) -> Bool {
    if case .ready = states[spec] { return true }
    return false
  }
}

/// A loaded engine and the queue of its requests.
final class EngineEntry: Sendable {
  let engine: any LoadedEngine
  let gate = GenerationGate()

  init(engine: any LoadedEngine) {
    self.engine = engine
  }
}

/// Failure messages per spec, readable without `await`.
final class LoadFailures: Sendable {
  private let messages = OSAllocatedUnfairLock<[EngineSpec: String]>(initialState: [:])

  func message(for spec: EngineSpec) -> String? {
    messages.withLock { $0[spec] }
  }

  func record(_ message: String, for spec: EngineSpec) {
    messages.withLock { $0[spec] = message }
  }

  func clear(_ spec: EngineSpec) {
    _ = messages.withLock { $0.removeValue(forKey: spec) }
  }
}

/// One generation at a time per engine; later requests wait in arrival order.
///
/// Whether one LiteRT-LM engine can run two conversations at once is not established, and a
/// second engine for the same file would double the memory, so requests take turns.
actor GenerationGate {
  private struct Waiter {
    let id: UInt64
    let continuation: CheckedContinuation<Void, any Error>
  }

  private var busy = false
  private var waiters: [Waiter] = []
  private var nextID: UInt64 = 0

  /// Waits for the turn. Throws `CancellationError` when `cancellation` fires first.
  func acquire(_ cancellation: RequestCancellation) async throws {
    if cancellation.isCancelled { throw CancellationError() }
    guard busy else {
      busy = true
      return
    }
    nextID &+= 1
    let id = nextID
    let token = cancellation.onCancel {
      Task { await self.abandon(id) }
    }
    defer {
      if let token { cancellation.removeHandler(token) }
    }
    try await withCheckedThrowingContinuation { continuation in
      waiters.append(Waiter(id: id, continuation: continuation))
    }
  }

  /// Hands the turn to the next waiter, or frees the gate.
  func release() {
    if waiters.isEmpty {
      busy = false
    } else {
      waiters.removeFirst().continuation.resume()
    }
  }

  /// The number of requests waiting (for tests).
  var waitingCount: Int { waiters.count }

  private func abandon(_ id: UInt64) {
    guard let index = waiters.firstIndex(where: { $0.id == id }) else { return }
    waiters.remove(at: index).continuation.resume(throwing: CancellationError())
  }
}

/// The cancellation of one request, shared by the caller's side (a cancelled task or a
/// stream whose reader went away) and the generation's side.
final class RequestCancellation: Sendable {
  private struct State {
    var cancelled = false
    var nextToken: UInt64 = 0
    var handlers: [UInt64: @Sendable () -> Void] = [:]
  }

  private let state = OSAllocatedUnfairLock(initialState: State())

  var isCancelled: Bool {
    state.withLock { $0.cancelled }
  }

  /// Marks the request cancelled and runs the handlers once.
  func cancel() {
    let handlers: [@Sendable () -> Void] = state.withLock { state in
      guard !state.cancelled else { return [] }
      state.cancelled = true
      let handlers = Array(state.handlers.values)
      state.handlers = [:]
      return handlers
    }
    for handler in handlers { handler() }
  }

  /// Runs `handler` on cancellation, now if the request is already cancelled.
  /// Returns a token for `removeHandler`, or `nil` when the handler already ran.
  @discardableResult
  func onCancel(_ handler: @escaping @Sendable () -> Void) -> UInt64? {
    let token: UInt64? = state.withLock { state in
      if state.cancelled { return nil }
      state.nextToken &+= 1
      state.handlers[state.nextToken] = handler
      return state.nextToken
    }
    if token == nil { handler() }
    return token
  }

  func removeHandler(_ token: UInt64) {
    _ = state.withLock { $0.handlers.removeValue(forKey: token) }
  }
}
