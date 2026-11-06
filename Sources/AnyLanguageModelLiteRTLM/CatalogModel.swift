import enum LiteRTFoundation.LiteRTModel

extension LiteRTLMLanguageModel {
  /// A model from swift-litert-lm's catalog, written as `.gemma4_E2B`.
  ///
  /// This type belongs to this package so that `.gemma4_E2B` resolves with only
  /// `import AnyLanguageModelLiteRTLM`, also in a target with `MemberImportVisibility` on
  /// (new Xcode 27 projects turn it on). Re-exporting swift-litert-lm's `LiteRTModel` did not
  /// make its cases visible there.
  public struct CatalogModel: Hashable, Sendable {
    let model: LiteRTModel

    /// Gemma 4 E2B, instruction-tuned (about 2.6 GB).
    public static let gemma4_E2B = CatalogModel(.gemma4_E2B)

    /// A case of swift-litert-lm's `LiteRTModel` catalog (needs `import LiteRTFoundation`).
    public init(_ model: LiteRTModel) {
      self.model = model
    }
  }
}
