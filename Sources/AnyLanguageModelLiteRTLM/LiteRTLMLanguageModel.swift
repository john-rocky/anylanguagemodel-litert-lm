import AnyLanguageModel
import Foundation
import enum LiteRTFoundation.LiteRTModel
import enum LiteRTLM.Backend
import os

/// A LiteRT-LM model for AnyLanguageModel's `LanguageModelSession`.
///
/// ```swift
/// let model = LiteRTLMLanguageModel(.gemma4_E2B)
/// try await model.load()  // downloads about 2.6 GB on the first run
/// let session = LanguageModelSession(model: model)
/// let response = try await session.respond(to: "Write a haiku about the sea.")
/// ```
///
/// The value is cheap: creating it does not load anything. The engine loads on `load()`,
/// `prewarm`, or the first request, once per model file for the whole process, and models with
/// the same file and settings share it. One request runs at a time per engine; a second request
/// waits for the first. Each request starts a new LiteRT-LM conversation from the session's
/// transcript.
///
/// `@Generable` output goes through LiteRT-LM's constrained decoding, so the reply is JSON
/// that matches the type's schema, with the keys in alphabetical order.
public struct LiteRTLMLanguageModel: LanguageModel {
  /// Why the model is not ready.
  public enum UnavailableReason: Equatable, Hashable, Sendable {
    /// No file exists at the model file URL.
    case modelFileMissing(URL)
    /// The catalog or Hugging Face model is not downloaded yet. A request downloads it.
    case notDownloaded
    /// The device has less RAM than the catalog model needs.
    case insufficientMemory(availableBytes: Int64, requiredBytes: Int64)
    /// The last attempt to load the model failed, with LiteRT-LM's message.
    /// The next request tries again.
    case loadFailed(String)
  }

  enum Source: Sendable {
    case catalog(LiteRTModel, storageDirectory: URL?, allowUnsafeMemory: Bool)
    case file(URL)
    case huggingFace(repo: String, fileName: String, revision: String, storageDirectory: URL?)
  }

  let source: Source
  let backend: Backend
  let maxTokens: Int
  let cacheDirectory: URL?
  let enableBenchmark: Bool
  let onDownloadProgress: (@Sendable (DownloadProgress) -> Void)?
  let store: EngineStore
  let physicalMemory: @Sendable () -> Int64

  // MARK: Initializers

  /// A model from swift-litert-lm's catalog, downloaded on first use into
  /// Application Support/LiteRTModels (the same file `LiteRTChat` uses).
  ///
  /// - Parameters:
  ///   - model: The catalog model, such as `.gemma4_E2B`.
  ///   - backend: Where the model runs. The Metal GPU engages only in a signed app on a
  ///     physical device; the iOS Simulator runs on the CPU.
  ///   - storageDirectory: Where to keep the downloaded file. Defaults to
  ///     Application Support/LiteRTModels.
  ///   - cacheDirectory: Where LiteRT-LM writes its GPU cache (about 0.8 GB for Gemma 4 E2B).
  ///     Defaults to the Caches directory, as `LiteRTChat` does.
  ///   - allowUnsafeMemory: Skip the check that the device has the RAM the model needs.
  ///   - enableBenchmark: Turn on LiteRT-LM's benchmark counters (a process-wide flag), so
  ///     each response's `usage` carries token counts and `usage.metadata` carries
  ///     `decodeTokensPerSecond`, `prefillTokensPerSecond` and `timeToFirstTokenSeconds`.
  ///   - onDownloadProgress: Called as the model file downloads.
  public init(
    _ model: CatalogModel,
    backend: Backend = .gpu,
    storageDirectory: URL? = nil,
    cacheDirectory: URL? = nil,
    allowUnsafeMemory: Bool = false,
    enableBenchmark: Bool = false,
    onDownloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
  ) {
    self.init(
      source: .catalog(
        model.model, storageDirectory: storageDirectory, allowUnsafeMemory: allowUnsafeMemory),
      backend: backend,
      maxTokens: model.model.defaultMaxTokens,
      cacheDirectory: cacheDirectory,
      enableBenchmark: enableBenchmark,
      onDownloadProgress: onDownloadProgress)
  }

  /// A `.litertlm` file already on disk: bundled, copied to the device, or fine-tuned.
  ///
  /// - Parameters:
  ///   - modelFileURL: The file's URL.
  ///   - maxTokens: The context size: prompt and reply tokens together.
  ///   - (Other parameters as in the catalog initializer.)
  public init(
    modelFileURL: URL,
    backend: Backend = .gpu,
    maxTokens: Int = 2048,
    cacheDirectory: URL? = nil,
    enableBenchmark: Bool = false
  ) {
    self.init(
      source: .file(modelFileURL),
      backend: backend,
      maxTokens: maxTokens,
      cacheDirectory: cacheDirectory,
      enableBenchmark: enableBenchmark,
      onDownloadProgress: nil)
  }

  /// A `.litertlm` file in a Hugging Face repo, downloaded on first use.
  ///
  /// - Parameters:
  ///   - huggingFaceRepo: The repo, such as `"litert-community/gemma-4-E2B-it-litert-lm"`.
  ///   - fileName: The `.litertlm` file in the repo.
  ///   - revision: The branch, tag or commit.
  ///   - (Other parameters as in the other initializers.)
  public init(
    huggingFaceRepo: String,
    fileName: String,
    revision: String = "main",
    backend: Backend = .gpu,
    storageDirectory: URL? = nil,
    maxTokens: Int = 2048,
    cacheDirectory: URL? = nil,
    enableBenchmark: Bool = false,
    onDownloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
  ) {
    self.init(
      source: .huggingFace(
        repo: huggingFaceRepo, fileName: fileName, revision: revision,
        storageDirectory: storageDirectory),
      backend: backend,
      maxTokens: maxTokens,
      cacheDirectory: cacheDirectory,
      enableBenchmark: enableBenchmark,
      onDownloadProgress: onDownloadProgress)
  }

  init(
    source: Source,
    backend: Backend,
    maxTokens: Int,
    cacheDirectory: URL?,
    enableBenchmark: Bool,
    onDownloadProgress: (@Sendable (DownloadProgress) -> Void)?,
    store: EngineStore = .shared,
    physicalMemory: @escaping @Sendable () -> Int64 = { Int64(ProcessInfo.processInfo.physicalMemory) }
  ) {
    self.source = source
    self.backend = backend
    self.maxTokens = maxTokens
    self.cacheDirectory = cacheDirectory
    self.enableBenchmark = enableBenchmark
    self.onDownloadProgress = onDownloadProgress
    self.store = store
    self.physicalMemory = physicalMemory
  }

  // MARK: Loading

  /// Where the model file is, or will be once downloaded.
  var modelFileURL: URL? {
    switch source {
    case .file(let url):
      return url
    case .catalog(let model, let storageDirectory, _):
      return (storageDirectory ?? ModelFiles.defaultStorageDirectory())?
        .appendingPathComponent(model.fileName)
    case .huggingFace(_, let fileName, _, let storageDirectory):
      return (storageDirectory ?? ModelFiles.defaultStorageDirectory())?
        .appendingPathComponent(fileName)
    }
  }

  var spec: EngineSpec? {
    modelFileURL.map { url in
      EngineSpec(
        modelPath: url.path,
        backend: backend,
        maxTokens: maxTokens,
        cacheDirectory: (cacheDirectory ?? ModelFiles.defaultCacheDirectory())?.path,
        enableBenchmark: enableBenchmark)
    }
  }

  /// Downloads the model file if needed, then loads the engine. Returns at once when the
  /// engine is loaded. Call it before showing a screen that should not wait for the first
  /// reply: a session request does the same work on first use.
  ///
  /// - Throws: `LiteRTLMLanguageModelError`.
  public func load() async throws {
    _ = try await engineEntry()
  }

  /// Releases the engine. Requests in flight finish first; the next request loads it again.
  public func unload() async {
    guard let spec else { return }
    await store.unload(spec)
  }

  func engineEntry() async throws -> EngineEntry {
    guard let url = modelFileURL, let spec else {
      throw LiteRTLMLanguageModelError.downloadFailed("The Application Support directory is not available.")
    }
    var fetch: (@Sendable () async throws -> Void)?
    switch source {
    case .file(let url):
      guard ModelFiles.fileExists(url) else {
        throw LiteRTLMLanguageModelError.modelFileMissing(url)
      }
    case .catalog(let model, let storageDirectory, let allowUnsafeMemory):
      let ram = physicalMemory()
      if !allowUnsafeMemory, ram < model.minimumDeviceRAM {
        throw LiteRTLMLanguageModelError.insufficientMemory(
          availableBytes: ram, requiredBytes: model.minimumDeviceRAM)
      }
      let progress = onDownloadProgress
      fetch = {
        try await ModelFiles.downloadCatalogModel(
          model, storageDirectory: storageDirectory, onProgress: progress)
      }
    case .huggingFace(let repo, let fileName, let revision, _):
      let progress = onDownloadProgress
      fetch = {
        try await ModelFiles.downloadFromHuggingFace(
          repo: repo, fileName: fileName, revision: revision, destination: url,
          onProgress: progress)
      }
    }
    do {
      return try await store.entry(for: spec, fetch: fetch)
    } catch let error as LiteRTLMLanguageModelError {
      throw error
    } catch is CancellationError {
      throw CancellationError()
    } catch {
      throw LiteRTLMLanguageModelError.loadFailed(ErrorMapping.describe(error))
    }
  }

  // MARK: LanguageModel

  public var availability: Availability<UnavailableReason> {
    switch source {
    case .file(let url):
      if !ModelFiles.fileExists(url) { return .unavailable(.modelFileMissing(url)) }
    case .catalog(let model, _, let allowUnsafeMemory):
      let ram = physicalMemory()
      if !allowUnsafeMemory, ram < model.minimumDeviceRAM {
        return .unavailable(
          .insufficientMemory(availableBytes: ram, requiredBytes: model.minimumDeviceRAM))
      }
      guard let url = modelFileURL, ModelFiles.fileExists(url) else {
        return .unavailable(.notDownloaded)
      }
    case .huggingFace:
      guard let url = modelFileURL, ModelFiles.fileExists(url) else {
        return .unavailable(.notDownloaded)
      }
    }
    if let spec, let message = store.failures.message(for: spec) {
      return .unavailable(.loadFailed(message))
    }
    return .available
  }

  /// Starts `load()` in the background. The prompt prefix is not used: each request starts a
  /// new conversation, so nothing prefilled now would carry over.
  public func prewarm(for session: LanguageModelSession, promptPrefix: Prompt? = nil) {
    let model = self
    Task { try? await model.load() }
  }

  public func respond<Content>(
    within session: LanguageModelSession,
    to prompt: Prompt,
    generating type: Content.Type,
    includeSchemaInPrompt: Bool,
    options: GenerationOptions
  ) async throws -> LanguageModelSession.Response<Content> where Content: Generable {
    let schema = Content.self == String.self ? nil : Content.generationSchema
    let result = try await generate(
      within: session, to: prompt, schema: schema,
      includeSchemaInPrompt: includeSchemaInPrompt, options: options)
    return try Self.response(result, structured: schema != nil, as: Content.self)
  }

  public func respond(
    within session: LanguageModelSession,
    to prompt: Prompt,
    schema: GenerationSchema,
    includeSchemaInPrompt: Bool,
    options: GenerationOptions
  ) async throws -> LanguageModelSession.Response<GeneratedContent> {
    let result = try await generate(
      within: session, to: prompt, schema: schema,
      includeSchemaInPrompt: includeSchemaInPrompt, options: options)
    return try Self.response(result, structured: true, as: GeneratedContent.self)
  }

  public func streamResponse<Content>(
    within session: LanguageModelSession,
    to prompt: Prompt,
    generating type: Content.Type,
    includeSchemaInPrompt: Bool,
    options: GenerationOptions
  ) -> sending LanguageModelSession.ResponseStream<Content> where Content: Generable {
    let schema = Content.self == String.self ? nil : Content.generationSchema
    return LanguageModelSession.ResponseStream(
      stream: snapshots(
        within: session, to: prompt, schema: schema,
        includeSchemaInPrompt: includeSchemaInPrompt, options: options, as: Content.self))
  }

  public func streamResponse(
    within session: LanguageModelSession,
    to prompt: Prompt,
    schema: GenerationSchema,
    includeSchemaInPrompt: Bool,
    options: GenerationOptions
  ) -> sending LanguageModelSession.ResponseStream<GeneratedContent> {
    LanguageModelSession.ResponseStream(
      stream: snapshots(
        within: session, to: prompt, schema: schema,
        includeSchemaInPrompt: includeSchemaInPrompt, options: options,
        as: GeneratedContent.self))
  }

  // MARK: Requests

  /// The request for the session's current state: messages, sampler, schema.
  func makeRequest(
    within session: LanguageModelSession,
    to prompt: Prompt,
    schema: GenerationSchema?,
    includeSchemaInPrompt: Bool,
    options: GenerationOptions
  ) throws -> GenerationRequest {
    let context = session.resolvedRequestContext()
    let plan = try TranscriptPlan(
      transcript: context.transcript, tools: context.tools,
      toolCallingMode: options.toolCallingMode, currentPrompt: prompt)
    let schemaJSON = try schema.map(StructuredOutput.schemaJSON(for:))
    return GenerationRequest(
      setup: ConversationSetup(
        systemMessage: plan.systemMessage,
        history: plan.history,
        sampler: SamplerSettings(options: options),
        constrained: schemaJSON != nil),
      message: StructuredOutput.message(
        prompt: plan.prompt, schemaJSON: schemaJSON, includeSchemaInPrompt: includeSchemaInPrompt),
      maxOutputTokens: maxOutputTokens(from: options),
      schemaJSON: schemaJSON)
  }

  func generate(
    within session: LanguageModelSession,
    to prompt: Prompt,
    schema: GenerationSchema?,
    includeSchemaInPrompt: Bool,
    options: GenerationOptions
  ) async throws -> GenerationResult {
    let request = try makeRequest(
      within: session, to: prompt, schema: schema,
      includeSchemaInPrompt: includeSchemaInPrompt, options: options)
    let cancellation = RequestCancellation()
    return try await withTaskCancellationHandler {
      let entry: EngineEntry
      do {
        entry = try await engineEntry()
      } catch {
        throw ErrorMapping.loadError(error)
      }
      return try await entry.run(request, streaming: false, cancellation: cancellation)
    } onCancel: {
      cancellation.cancel()
    }
  }

  func snapshots<Content: Generable>(
    within session: LanguageModelSession,
    to prompt: Prompt,
    schema: GenerationSchema?,
    includeSchemaInPrompt: Bool,
    options: GenerationOptions,
    as type: Content.Type
  ) -> AsyncThrowingStream<LanguageModelSession.ResponseStream<Content>.Snapshot, any Error> {
    let model = self
    let structured = schema != nil
    return AsyncThrowingStream { continuation in
      let cancellation = RequestCancellation()
      // A reader that goes away cancels the request. The task below is not cancelled itself:
      // it waits for the native generation to stop before the engine takes the next request.
      continuation.onTermination = { termination in
        if case .cancelled = termination { cancellation.cancel() }
      }
      Task {
        do {
          let request = try model.makeRequest(
            within: session, to: prompt, schema: schema,
            includeSchemaInPrompt: includeSchemaInPrompt, options: options)
          let entry: EngineEntry
          do {
            entry = try await model.engineEntry()
          } catch {
            throw ErrorMapping.loadError(error)
          }
          let text = TextAccumulator()
          let result = try await entry.run(request, streaming: true, cancellation: cancellation) {
            delta in
            let soFar = text.append(delta)
            if let raw = StructuredOutput.partialRaw(soFar, structured: structured),
              let partial = partiallyGenerated(Content.self, from: raw)
            {
              continuation.yield(.init(content: partial, rawContent: raw))
            }
          }
          // The last snapshot is the whole reply, decoded as a complete value, with the usage.
          let response = try Self.response(result, structured: structured, as: Content.self)
          continuation.yield(
            .init(
              content: response.content.asPartiallyGenerated(),
              rawContent: response.rawContent,
              usage: response.usage))
          continuation.finish()
        } catch {
          continuation.finish(throwing: error)
        }
      }
    }
  }

  static func response<Content: Generable>(
    _ result: GenerationResult, structured: Bool, as type: Content.Type
  ) throws -> LanguageModelSession.Response<Content> {
    let usage = Self.usage(result.benchmark)
    if structured {
      let decoded = try StructuredOutput.decode(result.text, as: Content.self)
      return .init(
        content: decoded.content, rawContent: decoded.raw, transcriptEntries: [], usage: usage)
    }
    let raw = GeneratedContent(result.text)
    return .init(content: try Content(raw), rawContent: raw, transcriptEntries: [], usage: usage)
  }

  /// The benchmark counters as AnyLanguageModel's usage. Zero counts when benchmarking is off.
  static func usage(_ benchmark: RuntimeBenchmark?) -> LanguageModelSession.Usage {
    guard let benchmark else {
      return .init(
        input: .init(totalTokenCount: 0, cachedTokenCount: 0),
        output: .init(totalTokenCount: 0, reasoningTokenCount: 0))
    }
    // The KV cache holds the whole prompt (system message and history included) and the
    // reply; the last prefill can be only part of the prompt.
    let input = benchmark.kvCacheTokens.map { max(0, $0 - benchmark.decodeTokens) }
      ?? benchmark.prefillTokens
    return .init(
      input: .init(totalTokenCount: input, cachedTokenCount: 0),
      output: .init(totalTokenCount: benchmark.decodeTokens, reasoningTokenCount: 0),
      metadata: [
        "decodeTokensPerSecond": benchmark.decodeTokensPerSecond,
        "prefillTokensPerSecond": benchmark.prefillTokensPerSecond,
        "timeToFirstTokenSeconds": benchmark.timeToFirstTokenSeconds,
      ])
  }
}

/// The reply text of a stream so far.
final class TextAccumulator: Sendable {
  private let text = OSAllocatedUnfairLock(initialState: "")

  /// Appends `delta` and returns the whole text.
  func append(_ delta: String) -> String {
    text.withLock { text in
      text += delta
      return text
    }
  }
}
