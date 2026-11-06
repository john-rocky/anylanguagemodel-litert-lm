import AnyLanguageModel
import Foundation
import Testing

@testable import AnyLanguageModelLiteRTLM

@Suite struct CancellationTests {
  @Test func cancellingARespondCancelsTheNativeGeneration() async throws {
    let setup = try FakeSetup([.waitForCancel(deltasBefore: []), .reply("Next")])
    let session = LanguageModelSession(model: setup.model)
    let task = Task { try await session.respond(to: "Count to 500.") }
    let conversation = try await firstConversation(setup)
    await conversation.started.wait()
    task.cancel()
    await #expect(throws: CancellationError.self) { _ = try await task.value }
    #expect(conversation.cancelCount.current == 1)
    // The next request gets a new conversation and answers.
    let next = try await LanguageModelSession(model: setup.model).respond(to: "Hello")
    #expect(next.content == "Next")
  }

  @Test func aReaderThatStopsCancelsTheNativeGeneration() async throws {
    let setup = try FakeSetup([.waitForCancel(deltasBefore: ["1\n", "2\n", "3\n"]), .reply("Next")])
    let session = LanguageModelSession(model: setup.model)
    var seen = 0
    for try await _ in session.streamResponse(to: "Count to 500.") {
      seen += 1
      if seen == 2 { break }
    }
    let conversation = try await firstConversation(setup)
    try await waitUntil { conversation.cancelCount.current == 1 }
    let next = try await LanguageModelSession(model: setup.model).respond(to: "Hello")
    #expect(next.content == "Next")
  }

  @Test func aCancelledStreamTaskEnds() async throws {
    let setup = try FakeSetup([.waitForCancel(deltasBefore: ["1\n", "2\n", "3\n", "4\n", "5\n", "6\n"]), .reply("Next")])
    let session = LanguageModelSession(model: setup.model)
    let counter = Guarded(0)
    let task = Task {
      for try await _ in session.streamResponse(to: "Count to 500.") {
        counter.withLock { $0 += 1 }
      }
    }
    try await waitUntil { counter.current >= 5 }
    task.cancel()
    _ = await task.result
    let conversation = try await firstConversation(setup)
    try await waitUntil { conversation.cancelCount.current == 1 }
    let next = try await LanguageModelSession(model: setup.model).respond(to: "Hello")
    #expect(next.content == "Next")
  }

  @Test func requestsRunOneAtATimeInOrder() async throws {
    let setup = try FakeSetup([.reply("A")])
    let model = setup.model
    try await model.load()
    try await withThrowingTaskGroup(of: String.self) { group in
      for index in 0..<4 {
        group.addTask { try await LanguageModelSession(model: model).respond(to: "Request \(index)").content }
      }
      for try await reply in group { #expect(reply == "A") }
    }
    #expect(setup.engine.maxInFlight.current == 1)
    #expect(setup.engine.conversations.current.count == 4)
  }

  @Test func aWaitingRequestCanBeCancelled() async throws {
    let setup = try FakeSetup([.waitForCancel(deltasBefore: []), .reply("Later")])
    let model = setup.model
    try await model.load()
    let first = Task { try await LanguageModelSession(model: model).respond(to: "First") }
    let conversation = try await firstConversation(setup)
    await conversation.started.wait()
    let second = Task { try await LanguageModelSession(model: model).respond(to: "Second") }
    let spec = try #require(model.spec)
    let entry = try await setup.store.entry(for: spec)
    try await waitUntil { await entry.gate.waitingCount == 1 }
    second.cancel()
    await #expect(throws: CancellationError.self) { _ = try await second.value }
    #expect(await entry.gate.waitingCount == 0)
    // The second request never reached the engine.
    #expect(setup.engine.conversations.current.count == 1)
    first.cancel()
    _ = await first.result
  }

  private func firstConversation(_ setup: FakeSetup) async throws -> FakeConversation {
    try await waitUntil { !setup.engine.conversations.current.isEmpty }
    return setup.engine.conversations.current[0]
  }
}

struct TimedOut: Error {}

/// Polls `condition` for up to 5 s.
func waitUntil(_ condition: @Sendable () async -> Bool) async throws {
  for _ in 0..<500 {
    if await condition() { return }
    try await Task.sleep(nanoseconds: 10_000_000)
  }
  throw TimedOut()
}
