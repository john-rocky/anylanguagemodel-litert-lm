import AnyLanguageModel
import Foundation

/// A session's transcript as the messages of a new LiteRT-LM conversation.
///
/// Each request starts a new conversation from the transcript: instructions become the system
/// message, earlier prompts and responses become the history, and the current prompt is the
/// message to send. The transcript stays the only history, so editing or restoring it in the
/// session takes effect on the next request.
struct TranscriptPlan: Equatable, Sendable {
  var systemMessage: String?
  var history: [HistoryMessage]
  var prompt: String

  /// - Parameters:
  ///   - transcript: The request's transcript. The session appends the current prompt before
  ///     it calls the model, so a trailing prompt entry is the current prompt.
  ///   - tools: The tools the session offers.
  ///   - toolCallingMode: The request's tool-calling mode; `.disallowed` lets a session with
  ///     tools through, because the model may not call them anyway.
  ///   - currentPrompt: The prompt the session passed, used when the transcript does not end
  ///     with it.
  init(
    transcript: Transcript,
    tools: [any Tool],
    toolCallingMode: GenerationOptions.ToolCallingMode?,
    currentPrompt: Prompt
  ) throws {
    if !tools.isEmpty, toolCallingMode != .disallowed {
      throw LiteRTLMLanguageModelError.toolsNotSupported
    }

    var entries = Array(transcript)
    var current: String?
    let lastIndex = entries.lastIndex { entry in
      if case .reasoning = entry { return false }
      return true
    }
    if let lastIndex, case .prompt(let prompt) = entries[lastIndex] {
      current = try Self.text(of: prompt.segments)
      entries.remove(at: lastIndex)
    }

    var system: [String] = []
    var history: [HistoryMessage] = []
    for entry in entries {
      switch entry {
      case .instructions(let instructions):
        let text = try Self.text(of: instructions.segments)
        if !text.isEmpty { system.append(text) }
      case .prompt(let prompt):
        let text = try Self.text(of: prompt.segments)
        if !text.isEmpty { history.append(HistoryMessage(role: .user, text: text)) }
      case .response(let response):
        let text = try Self.text(of: response.segments)
        if !text.isEmpty { history.append(HistoryMessage(role: .model, text: text)) }
      case .reasoning:
        // The model's earlier reasoning stays in the transcript for display only.
        continue
      case .toolCalls:
        throw LiteRTLMLanguageModelError.unsupportedContent("tool calls in the transcript")
      case .toolOutput:
        throw LiteRTLMLanguageModelError.unsupportedContent("tool outputs in the transcript")
      }
    }

    self.systemMessage = system.isEmpty ? nil : system.joined(separator: "\n\n")
    self.history = history
    self.prompt = current ?? currentPrompt.description
  }

  init(systemMessage: String?, history: [HistoryMessage], prompt: String) {
    self.systemMessage = systemMessage
    self.history = history
    self.prompt = prompt
  }

  private static func text(of segments: [Transcript.Segment]) throws -> String {
    try segments.map { segment -> String in
      switch segment {
      case .text(let text):
        return text.content
      case .structure(let structure):
        return structure.content.jsonString
      case .image:
        throw LiteRTLMLanguageModelError.unsupportedContent("image input")
      }
    }.joined(separator: "\n")
  }
}
