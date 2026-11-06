import AnyLanguageModel
import Foundation
import LiteRTLM
import Testing

@testable import AnyLanguageModelLiteRTLM

// The native strings below are the ones LiteRT-LM v0.19.0 printed for these failures.

@Suite struct ErrorMappingTests {
  private let constraint = LiteRTLMError.conversation(
    .invalidResponse(
      "litert_lm_conversation_send_message failed with status 13: INTERNAL: Failed to create LLGuidance "
        + "constraint: Reference segment '$defs' not found in '#/$defs/Missing'."))
  private let tooLong = LiteRTLMError.conversation(
    .invalidResponse(
      "INVALID_ARGUMENT: Input token ids are too long. Exceeding the maximum number of tokens allowed: 6009 >= 2048"))
  private let cancelled = LiteRTLMError.conversation(.invalidResponse("CANCELLED: Task cancelled"))

  @Test func aSchemaTheConstraintRefusesIsAnUnsupportedGuide() {
    let mapped = ErrorMapping.generationError(constraint, cancelled: false, constrained: true)
    guard case LanguageModelSession.GenerationError.unsupportedGuide(let context) = mapped else {
      Issue.record("got \(mapped)")
      return
    }
    #expect(context.debugDescription.contains("Failed to create LLGuidance constraint"))
  }

  @Test func theConstraintTextWithoutASchemaPassesThrough() {
    let mapped = ErrorMapping.generationError(constraint, cancelled: false, constrained: false)
    #expect(mapped as? LiteRTLMError == constraint)
  }

  @Test func aLongInputExceedsTheContextWindow() {
    let mapped = ErrorMapping.generationError(tooLong, cancelled: false, constrained: false)
    guard case LanguageModelSession.GenerationError.exceededContextWindowSize = mapped else {
      Issue.record("got \(mapped)")
      return
    }
  }

  @Test func aCancelledRequestIsACancellationError() {
    #expect(ErrorMapping.generationError(cancelled, cancelled: true, constrained: false) is CancellationError)
    #expect(ErrorMapping.generationError(CancellationError(), cancelled: false, constrained: true) is CancellationError)
  }

  @Test func otherErrorsPassThrough() {
    let other = LiteRTLMError.conversation(.notAlive)
    #expect(ErrorMapping.generationError(other, cancelled: false, constrained: true) as? LiteRTLMError == other)
  }

  @Test func aLoadFailureIsAssetsUnavailableWithTheNativeText() async throws {
    let native = FakeNativeError(
      message: "Failed to create engine: litert_lm_engine_create failed with status 3: "
        + "INVALID_ARGUMENT: Invalid magic number or failed to read")
    let setup = try FakeSetup(failures: [native])
    let session = LanguageModelSession(model: setup.model)
    do {
      _ = try await session.respond(to: "Hello")
      Issue.record("responded with a broken model")
    } catch LanguageModelSession.GenerationError.assetsUnavailable(let context) {
      #expect(context.debugDescription.contains("Invalid magic number"))
    }
  }

  @Test func loadThrowsTheBackendError() async throws {
    let native = FakeNativeError(message: "NOT_FOUND: model_signature_utils.cc:318")
    let setup = try FakeSetup(failures: [native])
    await #expect(throws: LiteRTLMLanguageModelError.loadFailed("NOT_FOUND: model_signature_utils.cc:318")) {
      try await setup.model.load()
    }
  }

  @Test func aSendFailureReachesTheCaller() async throws {
    let setup = try FakeSetup([
      .fail(FakeNativeError(message: "INVALID_ARGUMENT: Input token ids are too long.")), .reply("Hi"),
    ])
    let session = LanguageModelSession(model: setup.model)
    do {
      _ = try await session.respond(to: String(repeating: "word ", count: 10))
      Issue.record("no error")
    } catch LanguageModelSession.GenerationError.exceededContextWindowSize {
    }
    // The next request on the same model gets a new conversation and answers.
    let next = try await LanguageModelSession(model: setup.model).respond(to: "Hello")
    #expect(next.content == "Hi")
  }

  @Test func aStructuredStreamWithAnUnsupportedSchemaThrows() async throws {
    let setup = try FakeSetup([
      .fail(FakeNativeError(message: "Failed to create LLGuidance constraint: Reference segment '$defs' not found"))
    ])
    let schema = try JSONDecoder().decode(
      GenerationSchema.self,
      from: Data(##"{"type":"object","properties":{"a":{"$ref":"#/$defs/Missing"}},"required":["a"]}"##.utf8))
    let session = LanguageModelSession(model: setup.model)
    do {
      for try await _ in session.streamResponse(to: "Answer.", schema: schema) {}
      Issue.record("no error")
    } catch LanguageModelSession.GenerationError.unsupportedGuide {
    }
  }
}
