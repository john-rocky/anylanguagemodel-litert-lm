// The catalog enum and the backend enum appear in `LiteRTLMLanguageModel`'s initializers.
// Re-exporting them lets an app write `LiteRTLMLanguageModel(.gemma4_E2B)` and
// `backend: .gpu` without importing swift-litert-lm's modules. The name `LiteRTModel`
// itself still needs `import LiteRTFoundation`.

@_exported import enum LiteRTFoundation.LiteRTModel
@_exported import enum LiteRTLM.Backend
