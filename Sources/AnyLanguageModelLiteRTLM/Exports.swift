// `Backend` appears in `LiteRTLMLanguageModel`'s initializers. Re-exporting it lets an app
// write `backend: .gpu` or `.cpu()` without importing swift-litert-lm's modules, also with
// `MemberImportVisibility` on. The catalog parameter is this package's own type
// (`LiteRTLMLanguageModel.CatalogModel`) instead: re-exporting `LiteRTFoundation.LiteRTModel`
// the same way did not make `.gemma4_E2B` resolve with `MemberImportVisibility` on (Swift 6.4).

@_exported import enum LiteRTLM.Backend
