import AnyLanguageModel
import Foundation

/// Errors that `LiteRTLMLanguageModel` throws itself.
///
/// `load()` throws the loading cases. A request through `LanguageModelSession` reports the
/// same failures as `LanguageModelSession.GenerationError.assetsUnavailable`, with this
/// error's description in the context's `debugDescription`.
public enum LiteRTLMLanguageModelError: Error, LocalizedError, Equatable, Sendable {
  /// The session offers tools. This version of the backend does not call tools.
  case toolsNotSupported
  /// The transcript holds content this version of the backend does not send to the model,
  /// such as an image, a tool call, or a tool output.
  case unsupportedContent(String)
  /// No file exists at the model file URL.
  case modelFileMissing(URL)
  /// The device has less RAM than the catalog model needs.
  case insufficientMemory(availableBytes: Int64, requiredBytes: Int64)
  /// The model file could not be downloaded.
  case downloadFailed(String)
  /// LiteRT-LM could not load the model file.
  case loadFailed(String)

  public var errorDescription: String? {
    switch self {
    case .toolsNotSupported:
      return "LiteRTLMLanguageModel does not call tools yet. Create the session without tools, "
        + "or set GenerationOptions.toolCallingMode to .disallowed."
    case .unsupportedContent(let what):
      return "LiteRTLMLanguageModel does not send \(what) to the model yet."
    case .modelFileMissing(let url):
      return "No model file found at \(url.path)."
    case .insufficientMemory(let available, let required):
      let have = ByteCountFormatter.string(fromByteCount: available, countStyle: .memory)
      let need = ByteCountFormatter.string(fromByteCount: required, countStyle: .memory)
      return "This device has \(have) of RAM; this model needs at least \(need). "
        + "Pass allowUnsafeMemory: true to try anyway."
    case .downloadFailed(let details):
      return "The model file could not be downloaded: \(details)"
    case .loadFailed(let details):
      return "LiteRT-LM could not load the model: \(details)"
    }
  }
}

/// Maps errors from loading and generation to the errors a session caller sees.
enum ErrorMapping {
  /// The text of an error, for a description or a match.
  static func describe(_ error: any Error) -> String {
    if let localized = (error as? LocalizedError)?.errorDescription, !localized.isEmpty {
      return localized
    }
    return String(describing: error)
  }

  /// A loading failure, as `respond` and `streamResponse` report it.
  static func loadError(_ error: any Error) -> any Error {
    if error is CancellationError { return error }
    return LanguageModelSession.GenerationError.assetsUnavailable(
      .init(debugDescription: describe(error)))
  }

  /// A failure while the native runtime generates.
  ///
  /// The runtime reports every failure as `LiteRTLMError` with the native status text, so the
  /// cases are told apart by that text (the strings LiteRT-LM v0.19.0 prints).
  static func generationError(_ error: any Error, cancelled: Bool, constrained: Bool) -> any Error {
    if cancelled || error is CancellationError { return CancellationError() }
    let text = describe(error)
    if constrained, text.localizedCaseInsensitiveContains("constraint") {
      return LanguageModelSession.GenerationError.unsupportedGuide(
        .init(debugDescription: "LiteRT-LM could not use the response schema: \(text)"))
    }
    if text.contains("Input token ids are too long")
      || text.contains("Exceeding the maximum number of tokens")
    {
      return LanguageModelSession.GenerationError.exceededContextWindowSize(
        .init(debugDescription: text))
    }
    return error
  }

  /// Output that does not decode as the requested type.
  static func decodingError(_ details: String) -> any Error {
    LanguageModelSession.GenerationError.decodingFailure(.init(debugDescription: details))
  }
}
