import AnyLanguageModel
import AnyLanguageModelLiteRTLM
import Foundation

// Compiled the way an app would write it: no @testable and no swift-litert-lm import in this
// file. The catalog case and the backends come through this package's re-exports.
enum ClientImports {
  static func models(_ file: URL) -> [LiteRTLMLanguageModel] {
    [
      LiteRTLMLanguageModel(.gemma4_E2B, backend: .gpu),
      LiteRTLMLanguageModel(modelFileURL: file, backend: .cpu()),
    ]
  }
}
