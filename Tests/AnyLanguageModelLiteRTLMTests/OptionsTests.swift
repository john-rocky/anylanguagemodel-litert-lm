import AnyLanguageModel
import Foundation
import Testing

@testable import AnyLanguageModelLiteRTLM

@Suite struct OptionsTests {
  private func settings(_ options: GenerationOptions) -> SamplerSettings {
    SamplerSettings(options: options, randomSeed: { 1234 })
  }

  @Test func allNilIsTheDefaultSampler() {
    #expect(settings(GenerationOptions()) == SamplerSettings(topK: 40, topP: 0.95, temperature: 0.8, seed: 1234))
  }

  @Test func temperatureOnly() {
    #expect(
      settings(GenerationOptions(temperature: 0.3))
        == SamplerSettings(topK: 40, topP: 0.95, temperature: 0.3, seed: 1234))
    #expect(settings(GenerationOptions(temperature: -1)).temperature == 0)
  }

  @Test func greedy() {
    #expect(
      settings(GenerationOptions(sampling: .greedy, temperature: 0.9))
        == SamplerSettings(topK: 1, topP: 1, temperature: 0, seed: 0))
  }

  @Test func topKWithSeed() {
    #expect(
      settings(GenerationOptions(sampling: .random(top: 20, seed: 7)))
        == SamplerSettings(topK: 20, topP: 1, temperature: 0.8, seed: 7))
    #expect(
      settings(GenerationOptions(sampling: .random(top: 20), temperature: 0.5))
        == SamplerSettings(topK: 20, topP: 1, temperature: 0.5, seed: 1234))
    #expect(settings(GenerationOptions(sampling: .random(top: 0))).topK == 1)
    // Past 64, LiteRT-LM rebuilds its GPU sampler with code that grows with k.
    #expect(settings(GenerationOptions(sampling: .random(top: 64))).topK == 64)
    #expect(settings(GenerationOptions(sampling: .random(top: 65))).topK == 64)
    #expect(settings(GenerationOptions(sampling: .random(top: .max))).topK == 64)
  }

  @Test func nucleus() {
    #expect(
      settings(GenerationOptions(sampling: .random(probabilityThreshold: 0.9)))
        == SamplerSettings(topK: 64, topP: 0.9, temperature: 0.8, seed: 1234))
    #expect(
      settings(GenerationOptions(sampling: .random(probabilityThreshold: 0.5, seed: 3))).seed == 3)
  }

  @Test func largeSeedsStayInInt32() {
    let seed = settings(GenerationOptions(sampling: .random(top: 40, seed: .max))).seed
    #expect(seed == Int(Int32.max))
    #expect(SamplerSettings.seed(from: 0x1_0000_0005) == 5)
    #expect(SamplerSettings.randomSeed() >= 0)
    #expect(SamplerSettings.randomSeed() <= Int(Int32.max))
  }

  @Test func maximumResponseTokens() {
    #expect(maxOutputTokens(from: GenerationOptions()) == nil)
    #expect(maxOutputTokens(from: GenerationOptions(maximumResponseTokens: 8)) == 8)
    #expect(maxOutputTokens(from: GenerationOptions(maximumResponseTokens: 0)) == 1)
    #expect(maxOutputTokens(from: GenerationOptions(maximumResponseTokens: .max)) == Int(Int32.max))
  }

  @Test func samplingIsReadFromTheTranscriptCoding() {
    #expect(SamplingChoice(reading: .greedy) == .greedy)
    #expect(SamplingChoice(reading: .random(top: 40, seed: 7)) == .topK(40, seed: 7))
    #expect(SamplingChoice(reading: .random(top: 40)) == .topK(40, seed: nil))
    #expect(SamplingChoice(reading: .random(probabilityThreshold: 0.9)) == .nucleus(0.9, seed: nil))
    #expect(
      SamplingChoice(reading: .random(probabilityThreshold: 0.25, seed: .max)) == .nucleus(0.25, seed: .max))
  }

  /// Pins the Codable form the backend reads. If AnyLanguageModel changes it, this fails
  /// before requests silently fall back to the default sampler.
  @Test func theTranscriptCodingKeepsItsShape() throws {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    func coded(_ options: GenerationOptions) throws -> String {
      String(decoding: try encoder.encode(Transcript.Prompt(id: "p", segments: [], options: options)), as: UTF8.self)
    }
    #expect(
      try coded(GenerationOptions(sampling: .random(top: 40, seed: 7), temperature: 0.5))
        .contains(#""options":{"sampling":{"mode":{"topK":{"_0":40,"seed":7}}},"temperature":0.5}"#))
    #expect(
      try coded(GenerationOptions(sampling: .random(probabilityThreshold: 0.9)))
        .contains(#""sampling":{"mode":{"nucleus":{"_0":0.9}}}"#))
    #expect(try coded(GenerationOptions(sampling: .greedy)).contains(#""sampling":{"mode":{"greedy":{}}}"#))
  }

  @Test func aRequestCarriesTheOptions() async throws {
    let setup = try FakeSetup([.reply("Hi")])
    let session = LanguageModelSession(model: setup.model)
    _ = try await session.respond(
      to: "Hello", options: GenerationOptions(sampling: .random(top: 5, seed: 9), temperature: 0.2, maximumResponseTokens: 16))
    #expect(setup.engine.lastSetup?.sampler == SamplerSettings(topK: 5, topP: 1, temperature: 0.2, seed: 9))
    #expect(setup.engine.lastSent?.maxOutputTokens == 16)
  }
}
