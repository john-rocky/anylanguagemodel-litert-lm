import AnyLanguageModel
import Foundation
import Testing

@testable import AnyLanguageModelLiteRTLM

@Suite struct AvailabilityTests {
  private func temporaryDirectory() throws -> URL {
    let directory = FileManager.default.temporaryDirectory
      .appendingPathComponent("alm-litertlm-availability-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    return directory
  }

  @Test func aMissingFile() async throws {
    let missing = try temporaryDirectory().appendingPathComponent("missing.litertlm")
    let model = LiteRTLMLanguageModel(modelFileURL: missing)
    #expect(model.availability == .unavailable(.modelFileMissing(missing)))
    await #expect(throws: LiteRTLMLanguageModelError.modelFileMissing(missing)) {
      try await model.load()
    }
    let session = LanguageModelSession(model: model)
    do {
      _ = try await session.respond(to: "Hello")
      Issue.record("responded without a model file")
    } catch LanguageModelSession.GenerationError.assetsUnavailable(let context) {
      #expect(context.debugDescription.contains("No model file found"))
    }
  }

  @Test func notEnoughRAM() throws {
    let storage = try temporaryDirectory()
    let small = LiteRTLMLanguageModel(
      source: .catalog(.gemma4_E2B, storageDirectory: storage, allowUnsafeMemory: false),
      backend: .gpu, maxTokens: 2048, cacheDirectory: storage, enableBenchmark: false,
      onDownloadProgress: nil, store: EngineStore(loader: FakeLoader(engine: FakeEngine([.reply("x")]))),
      physicalMemory: { 4_000_000_000 })
    #expect(
      small.availability
        == .unavailable(.insufficientMemory(availableBytes: 4_000_000_000, requiredBytes: 7_000_000_000)))
    let allowed = LiteRTLMLanguageModel(
      source: .catalog(.gemma4_E2B, storageDirectory: storage, allowUnsafeMemory: true),
      backend: .gpu, maxTokens: 2048, cacheDirectory: storage, enableBenchmark: false,
      onDownloadProgress: nil, physicalMemory: { 4_000_000_000 })
    #expect(allowed.availability == .unavailable(.notDownloaded))
  }

  @Test func aCatalogModelNotDownloadedYet() throws {
    let storage = try temporaryDirectory()
    let model = LiteRTLMLanguageModel(
      .gemma4_E2B, storageDirectory: storage, allowUnsafeMemory: true)
    #expect(model.availability == .unavailable(.notDownloaded))
    let hub = LiteRTLMLanguageModel(
      huggingFaceRepo: "litert-community/gemma-4-E2B-it-litert-lm", fileName: "gemma-4-E2B-it.litertlm",
      storageDirectory: storage)
    #expect(hub.availability == .unavailable(.notDownloaded))
    #expect(hub.modelFileURL == storage.appendingPathComponent("gemma-4-E2B-it.litertlm"))
  }

  @Test func aFailedLoadIsReportedThenRetried() async throws {
    let native = FakeNativeError(message: "INVALID_ARGUMENT: Invalid magic number or failed to read")
    let setup = try FakeSetup([.reply("Hi")], failures: [native])
    #expect(setup.model.availability == .available)
    await #expect(throws: LiteRTLMLanguageModelError.self) { try await setup.model.load() }
    #expect(setup.model.availability == .unavailable(.loadFailed(native.message)))
    // The next attempt loads again, and a load that works clears the failure.
    try await setup.model.load()
    #expect(setup.model.availability == .available)
    #expect(setup.loader.loadCount.current == 2)
  }

  @Test func loadsAreSingleFlight() async throws {
    let setup = try FakeSetup([.reply("Hi")], loadDelayNanoseconds: 200_000_000)
    async let first: Void = setup.model.load()
    async let second: Void = setup.model.load()
    async let third: Void = setup.model.load()
    _ = try await (first, second, third)
    #expect(setup.loader.loadCount.current == 1)
    let response = try await LanguageModelSession(model: setup.model).respond(to: "Hello")
    #expect(response.content == "Hi")
    #expect(setup.loader.loadCount.current == 1)
  }

  @Test func unloadDropsTheEngine() async throws {
    let setup = try FakeSetup([.reply("Hi")])
    try await setup.model.load()
    let spec = try #require(setup.model.spec)
    #expect(await setup.store.isLoaded(spec))
    await setup.model.unload()
    #expect(await setup.store.isLoaded(spec) == false)
    try await setup.model.load()
    #expect(setup.loader.loadCount.current == 2)
  }

  @Test func theEngineSpecFollowsTheInitializer() throws {
    let cache = try temporaryDirectory()
    let file = cache.appendingPathComponent("model.litertlm")
    let model = LiteRTLMLanguageModel(
      modelFileURL: file, backend: .cpu(), maxTokens: 1024, cacheDirectory: cache, enableBenchmark: true)
    #expect(
      model.spec
        == EngineSpec(
          modelPath: file.path, backend: .cpu(), maxTokens: 1024, cacheDirectory: cache.path,
          enableBenchmark: true))
    let catalog = LiteRTLMLanguageModel(.gemma4_E2B)
    #expect(catalog.spec?.maxTokens == 2048)
    #expect(catalog.spec?.backend == .gpu)
    #expect(catalog.spec?.modelPath.hasSuffix("LiteRTModels/gemma-4-E2B-it.litertlm") == true)
  }
}
