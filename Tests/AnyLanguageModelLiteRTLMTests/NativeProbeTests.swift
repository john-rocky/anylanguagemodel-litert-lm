import Foundation
import LiteRTLM
import Testing

// Calls LiteRT-LM directly, without AnyLanguageModel or this package's backend, to see how
// `maxOutputTokens` behaves with and without a response format. Skipped unless
// LITERT_NATIVE_PROBE lists probes ("noschema", "schema") and LITERT_TEST_MODEL points at
// a model file. Lines on stdout start with "NP:".

private let environment = ProcessInfo.processInfo.environment

@Suite(
  .serialized,
  .enabled(
    if: (environment["LITERT_NATIVE_PROBE"] != nil || environment["LITERT_NATIVE_SWEEP"] != nil)
      && environment["LITERT_TEST_MODEL"] != nil))
struct NativeProbeTests {
  @Test func maxOutputTokens() async throws {
    guard environment["LITERT_NATIVE_PROBE"] != nil else { return }
    let probes = (environment["LITERT_NATIVE_PROBE"] ?? "").split(separator: ",").map(String.init)
    let modelPath = environment["LITERT_TEST_MODEL"] ?? ""
    let cachePath = environment["LITERT_TEST_CACHE"]
    let maxOutputTokens = Int(environment["LITERT_NATIVE_PROBE_MAX_TOKENS"] ?? "") ?? 8
    let limit = UInt64(Double(environment["LITERT_TEST_MEMORY_LIMIT_GB"] ?? "") ?? 10) * 1_073_741_824
    let sampleFile = environment["LITERT_TEST_LOG_DIR"].map { "\($0)/native_probe_stack.txt" }
    MemoryGuard.start(limitBytes: limit, sampleFile: sampleFile, prefix: "NP:")

    let schema = try cafeReviewSchemaJSON()
    let prompt = "Invent a small cafe and review it.\n\nReply in JSON with the keys hasWifi, name, price and rating."
    note("start model=\(modelPath) maxOutputTokens=\(maxOutputTokens) probes=\(probes)")
    note("schema \(schema)")

    MemoryGuard.currentStep.withLock { $0 = "engine load" }
    let engine = Engine(
      engineConfig: try EngineConfig(
        modelPath: modelPath, backend: .gpu, maxNumTokens: 2048, cacheDir: cachePath))
    try await engine.initialize()
    note("engine loaded; footprint \(MemoryGuard.gigabytes(MemoryGuard.footprintBytes()))")

    for probe in probes {
      MemoryGuard.currentStep.withLock { $0 = "probe \(probe)" }
      let constrained = probe == "schema"
      let config = ConversationConfig(
        samplerConfig: try SamplerConfig(topK: 40, topP: 0.95, temperature: 0.8, seed: 7),
        automaticToolCalling: false,
        enableResponseFormat: constrained)
      let conversation = try await engine.createConversation(with: config)
      let start = Date()
      do {
        let reply = try await conversation.sendMessage(
          Message(prompt), maxOutputTokens: maxOutputTokens,
          responseFormat: constrained ? try ResponseFormat.json(schema: schema) : nil)
        note(
          "\(probe): returned in \(String(format: "%.1f", Date().timeIntervalSince(start))) s: "
            + "\(reply.toString.debugDescription); footprint \(MemoryGuard.gigabytes(MemoryGuard.footprintBytes()))")
      } catch {
        note(
          "\(probe): threw in \(String(format: "%.1f", Date().timeIntervalSince(start))) s: \(error); "
            + "footprint \(MemoryGuard.gigabytes(MemoryGuard.footprintBytes()))")
      }
      withExtendedLifetime(conversation) {}
    }
    note("DONE")
  }

  /// LITERT_NATIVE_SWEEP="<first>-<last>" cut points × LITERT_NATIVE_SEEDS="<first>-<last>" seeds,
  /// each a new constrained conversation with `maxOutputTokens` set to the cut point.
  @Test func maxOutputTokensSweep() async throws {
    guard let sweep = environment["LITERT_NATIVE_SWEEP"] else { return }
    let cuts = range(sweep, default: 1...24)
    let seeds = range(environment["LITERT_NATIVE_SEEDS"], default: 1...3)
    let modelPath = environment["LITERT_TEST_MODEL"] ?? ""
    let limit = UInt64(Double(environment["LITERT_TEST_MEMORY_LIMIT_GB"] ?? "") ?? 10) * 1_073_741_824
    let sampleFile = environment["LITERT_TEST_LOG_DIR"].map { "\($0)/native_sweep_stack.txt" }
    MemoryGuard.start(limitBytes: limit, sampleFile: sampleFile, prefix: "NP:")
    let schema = try cafeReviewSchemaJSON()
    let prompt = "Invent a small cafe and review it.\n\nReply in JSON with the keys hasWifi, name, price and rating."
    note("sweep start cuts=\(cuts) seeds=\(seeds)")
    MemoryGuard.currentStep.withLock { $0 = "engine load" }
    let engine = Engine(
      engineConfig: try EngineConfig(
        modelPath: modelPath, backend: .gpu, maxNumTokens: 2048, cacheDir: environment["LITERT_TEST_CACHE"]))
    try await engine.initialize()
    for seed in seeds {
      for cut in cuts {
        MemoryGuard.currentStep.withLock { $0 = "sweep seed=\(seed) maxOutputTokens=\(cut)" }
        let conversation = try await engine.createConversation(
          with: ConversationConfig(
            samplerConfig: try SamplerConfig(topK: 40, topP: 0.95, temperature: 0.8, seed: seed),
            automaticToolCalling: false, enableResponseFormat: true))
        let start = Date()
        let outcome: String
        do {
          let reply = try await conversation.sendMessage(
            Message(prompt), maxOutputTokens: cut, responseFormat: try ResponseFormat.json(schema: schema))
          outcome = "returned \(reply.toString.debugDescription)"
        } catch {
          outcome = "threw \(error)"
        }
        note(
          "sweep seed=\(seed) max=\(cut) \(String(format: "%.2f", Date().timeIntervalSince(start))) s "
            + "footprint \(MemoryGuard.gigabytes(MemoryGuard.footprintBytes())): \(outcome)")
        withExtendedLifetime(conversation) {}
      }
    }
    note("sweep DONE")
  }

  private func range(_ text: String?, default fallback: ClosedRange<Int>) -> ClosedRange<Int> {
    guard let text, text != "1" else { return fallback }
    let parts = text.split(separator: "-").compactMap { Int($0) }
    guard parts.count == 2, parts[0] <= parts[1] else { return fallback }
    return parts[0]...parts[1]
  }

  private func note(_ line: String) {
    print("NP: \(line)")
    fflush(stdout)
  }
}
