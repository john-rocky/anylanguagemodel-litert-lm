import AnyLanguageModel
import Foundation
import os

// GenerationOptions → LiteRT-LM's sampler and output limit.
//
//   all nil                         topK 40, topP 0.95, temperature 0.8, a random seed
//   temperature only                the same with that temperature
//   sampling .greedy                topK 1, topP 1, temperature 0
//   .random(top: k, seed:)          topK k (at most 64), topP 1
//   .random(probabilityThreshold:)  topP p within topK 64
//   maximumResponseTokens           the request's maxOutputTokens
//
// topK stops at 64, the top-k capacity LiteRT-LM gives its GPU sampler by default
// (v0.19.0, runtime/executor/llm_executor_settings.cc). A larger k makes LiteRT-LM rebuild the
// GPU sampler with code that grows with k: with k = Int32.max the process grew past 10 GB
// within seconds on a Mac (2026-11-07).
//
// `SamplingMode` keeps k, p and the seed internal; `== .greedy` is the only public check.
// The other values are read from `Transcript.Prompt`'s Codable form, which AnyLanguageModel
// keeps stable so that saved transcripts load. When that form cannot be read, the request
// uses the default sampler and logs one line.

extension SamplerSettings {
  static let defaultTopK = 40
  /// The largest topK the backend passes on (see the note at the top of this file).
  static let maximumTopK = 64
  static let defaultTopP: Float = 0.95
  static let defaultTemperature: Float = 0.8

  /// A seed in the range LiteRT-LM's sampler takes (a non-negative `Int32`).
  static func randomSeed() -> Int {
    Int.random(in: 0...Int(Int32.max))
  }

  init(options: GenerationOptions, randomSeed: () -> Int = SamplerSettings.randomSeed) {
    let temperature = options.temperature.map { Float(max(0, $0)) }
    switch options.sampling.map(SamplingChoice.init(reading:)) {
    case nil:
      self.init(
        topK: Self.defaultTopK, topP: Self.defaultTopP,
        temperature: temperature ?? Self.defaultTemperature, seed: randomSeed())
    case .greedy?:
      self.init(topK: 1, topP: 1, temperature: 0, seed: 0)
    case .topK(let k, let seed)?:
      self.init(
        topK: Self.clampedTopK(k), topP: 1,
        temperature: temperature ?? Self.defaultTemperature,
        seed: seed.map(Self.seed(from:)) ?? randomSeed())
    case .nucleus(let p, let seed)?:
      self.init(
        topK: Self.maximumTopK, topP: Float(min(1, max(0, p))),
        temperature: temperature ?? Self.defaultTemperature,
        seed: seed.map(Self.seed(from:)) ?? randomSeed())
    case .unreadable?:
      optionsLogger.notice(
        "Could not read GenerationOptions.sampling; this request uses the default sampler.")
      self.init(
        topK: Self.defaultTopK, topP: Self.defaultTopP,
        temperature: temperature ?? Self.defaultTemperature, seed: randomSeed())
    }
  }

  /// LiteRT-LM converts the seed with `Int32(_:)`, which traps past `Int32.max`; a negative
  /// seed is avoided too. The low 31 bits keep the same seed giving the same sampler.
  static func seed(from value: UInt64) -> Int {
    Int(value & 0x7FFF_FFFF)
  }

  static func clampedTopK(_ k: Int) -> Int {
    min(max(1, k), maximumTopK)
  }
}

/// The request's output limit for LiteRT-LM (`maxOutputTokens`, an `Int32` there).
func maxOutputTokens(from options: GenerationOptions) -> Int? {
  options.maximumResponseTokens.map { min(max(1, $0), Int(Int32.max)) }
}

/// What `GenerationOptions.SamplingMode` holds.
enum SamplingChoice: Equatable {
  case greedy
  case topK(Int, seed: UInt64?)
  case nucleus(Double, seed: UInt64?)
  case unreadable

  init(reading mode: GenerationOptions.SamplingMode) {
    if mode == .greedy {
      self = .greedy
      return
    }
    let prompt = Transcript.Prompt(id: "", segments: [], options: GenerationOptions(sampling: mode))
    guard let data = try? JSONEncoder().encode(prompt),
      let coded = try? JSONDecoder().decode(CodedPrompt.self, from: data),
      let coding = coded.options.sampling?.mode
    else {
      self = .unreadable
      return
    }
    switch coding {
    case .greedy: self = .greedy
    case .topK(let k, let seed): self = .topK(k, seed: seed)
    case .nucleus(let p, let seed): self = .nucleus(p, seed: seed)
    }
  }
}

/// The part of `Transcript.Prompt`'s Codable form that holds the sampling mode:
/// `{"options":{"sampling":{"mode":{"topK":{"_0":40,"seed":7}}}}}`.
private struct CodedPrompt: Decodable {
  struct Options: Decodable {
    var sampling: Sampling?
  }

  struct Sampling: Decodable {
    var mode: Mode
  }

  /// Mirrors `GenerationOptions.SamplingMode.Mode`, so the synthesized coding matches.
  enum Mode: Decodable {
    case greedy
    case topK(Int, seed: UInt64?)
    case nucleus(Double, seed: UInt64?)
  }

  var options: Options
}

private let optionsLogger = Logger(subsystem: "AnyLanguageModelLiteRTLM", category: "Options")
