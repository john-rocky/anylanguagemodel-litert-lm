import Foundation

/// One request as the runtime sees it.
struct GenerationRequest: Equatable, Sendable {
  var setup: ConversationSetup
  /// The current user message, with the key line appended for structured output.
  var message: String
  var maxOutputTokens: Int?
  /// The JSON schema for constrained decoding, or `nil` for free text.
  var schemaJSON: String?
}

struct GenerationResult: Sendable {
  var text: String
  var benchmark: RuntimeBenchmark?
}

extension EngineEntry {
  /// Runs one request in a new conversation, after the requests before it.
  ///
  /// Each request gets its own conversation, so a request that throws or is cancelled leaves
  /// nothing behind for the next one. The call returns only after the native generation
  /// ended, so the next request never starts while the engine still generates.
  ///
  /// - Parameter onDelta: Called with each text delta when `streaming` is true.
  func run(
    _ request: GenerationRequest,
    streaming: Bool,
    cancellation: RequestCancellation,
    onDelta: @escaping @Sendable (String) -> Void = { _ in }
  ) async throws -> GenerationResult {
    try await gate.acquire(cancellation)
    do {
      let result = try await runInTurn(
        request, streaming: streaming, cancellation: cancellation, onDelta: onDelta)
      await gate.release()
      return result
    } catch {
      await gate.release()
      throw error
    }
  }

  private func runInTurn(
    _ request: GenerationRequest,
    streaming: Bool,
    cancellation: RequestCancellation,
    onDelta: @escaping @Sendable (String) -> Void
  ) async throws -> GenerationResult {
    let conversation = try await engine.makeConversation(request.setup)
    let token = cancellation.onCancel { conversation.cancel() }
    defer {
      if let token { cancellation.removeHandler(token) }
      withExtendedLifetime(conversation) {}
    }
    if cancellation.isCancelled { throw CancellationError() }

    var text = ""
    do {
      if streaming {
        let deltas = conversation.stream(
          request.message, maxOutputTokens: request.maxOutputTokens, jsonSchema: request.schemaJSON)
        // This task is never cancelled from outside (cancellation goes through
        // `conversation.cancel()`), so the loop reads the stream to the native end.
        for try await delta in deltas {
          text += delta
          if !cancellation.isCancelled { onDelta(delta) }
        }
      } else {
        text = try await conversation.send(
          request.message, maxOutputTokens: request.maxOutputTokens, jsonSchema: request.schemaJSON)
      }
    } catch {
      throw ErrorMapping.generationError(
        error, cancelled: cancellation.isCancelled, constrained: request.schemaJSON != nil)
    }
    if cancellation.isCancelled { throw CancellationError() }
    return GenerationResult(text: text, benchmark: conversation.benchmark())
  }
}
