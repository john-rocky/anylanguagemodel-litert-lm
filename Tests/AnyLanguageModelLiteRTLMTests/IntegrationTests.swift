import AnyLanguageModel
import AnyLanguageModelLiteRTLM
import Foundation
import Testing

// Runs a real `.litertlm` through `LanguageModelSession`. Skipped unless LITERT_TEST_MODEL
// points at a model file (Gemma 4 E2B: gemma-4-E2B-it.litertlm).
//
//   LITERT_TEST_MODEL=/path/gemma-4-E2B-it.litertlm \
//   LITERT_TEST_CACHE=/path/cache LITERT_TEST_LOG_DIR=/path/logs \
//   swift test --filter IntegrationTests
//
// LITERT_TEST_CACHE is where LiteRT-LM writes its GPU cache (about 0.8 GB; defaults to a
// directory next to the model). LITERT_TEST_LOG_DIR keeps one log per leg. LITERT_TEST_LEGS
// runs only the named legs (a comma list). LITERT_TEST_MEMORY_LIMIT_GB (default 10) stops the
// process, with a stack sample, when its footprint passes the limit. Every result line on
// stdout starts with "IT:".

private let environment = ProcessInfo.processInfo.environment

@Suite(.serialized, .enabled(if: environment["LITERT_TEST_MODEL"] != nil))
struct IntegrationTests {
  private let modelURL = URL(fileURLWithPath: environment["LITERT_TEST_MODEL"] ?? "/missing.litertlm")
  private var cacheURL: URL {
    environment["LITERT_TEST_CACHE"].map { URL(fileURLWithPath: $0, isDirectory: true) }
      ?? modelURL.deletingLastPathComponent().appendingPathComponent("litertlm-cache", isDirectory: true)
  }
  private var logDirectory: URL? {
    environment["LITERT_TEST_LOG_DIR"].map { URL(fileURLWithPath: $0, isDirectory: true) }
  }

  private let textPrompt = "Write a haiku about the sea."
  private let flatPrompt = "Invent a small cafe and review it."
  private let nestedPrompt = "Plan a one-day trip in Kyoto."
  private let nextPrompt = "Reply with one short sentence about the sea."

  /// One test, so the legs run in the order the run was registered in.
  @Test func legs() async throws {
    try FileManager.default.createDirectory(at: cacheURL, withIntermediateDirectories: true)
    if let logDirectory {
      try FileManager.default.createDirectory(at: logDirectory, withIntermediateDirectories: true)
    }
    let limit = UInt64(Double(environment["LITERT_TEST_MEMORY_LIMIT_GB"] ?? "") ?? 10) * 1_073_741_824
    MemoryGuard.start(
      limitBytes: limit, sampleFile: logDirectory.map { $0.appendingPathComponent("memory_stack.txt").path },
      prefix: "IT:")
    let model = LiteRTLMLanguageModel(
      modelFileURL: modelURL, cacheDirectory: cacheURL, enableBenchmark: true)
    say("start model=\(modelURL.path) os=\(ProcessInfo.processInfo.operatingSystemVersionString)")
    say("memory limit \(MemoryGuard.gigabytes(limit)); legs \(selectedLegs.map { $0.sorted().joined(separator: ",") } ?? "all")")
    say("availability before load: \(model.availability)")

    var counted: [(String, Bool)] = []
    func count(_ name: String, _ passed: Bool?) {
      if let passed { counted.append((name, passed)) }
    }

    // Counted: respond 1, stream 1, flat 3, nested 3.
    count("respond", await leg("respond") { log in try await respondLeg(model, log) })
    count("stream", await leg("stream") { log in try await streamLeg(model, log) })
    for run in 1...3 {
      count("flat \(run)", await leg("flat_\(run)") { log in try await flatLeg(model, log) })
    }
    for run in 1...3 {
      count("nested \(run)", await leg("nested_\(run)") { log in try await nestedLeg(model, log) })
    }

    // Recorded, not counted.
    _ = await leg("flat_noschema") { log in try await flatLeg(model, log, includeSchemaInPrompt: false) }
    _ = await leg("nested_noschema") { log in try await nestedLeg(model, log, includeSchemaInPrompt: false) }
    _ = await leg("flat_stream") { log in try await flatStreamLeg(model, log) }
    _ = await leg("nested_stream") { log in try await nestedStreamLeg(model, log) }
    _ = await leg("two_sessions") { log in try await twoSessionsLeg(model, log) }
    _ = await leg("greedy_twice") { log in try await greedyLeg(model, log) }
    _ = await leg("max_tokens_8_flat") { log in try await maxTokensFlatLeg(model, log) }
    _ = await leg("max_tokens_8_text") { log in try await maxTokensTextLeg(model, log) }
    _ = await leg("nucleus") { log in try await nucleusLeg(model, log) }
    _ = await leg("cancel_respond") { log in try await cancelRespondLeg(model, log) }

    // Counted: the failure paths.
    let scratch = cacheURL.appendingPathComponent("failure-paths-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: scratch) }
    count("(a)", await leg("fail_a") { log in try await missingFileLeg(model, scratch, log) })
    count("(b1)", await leg("fail_b1") { log in try await randomBytesLeg(model, scratch, log) })
    count("(b2)", await leg("fail_b2") { log in try await truncatedLeg(model, scratch, log) })
    count("(b3)", await leg("fail_b3") { log in try await directoryLeg(model, scratch, log) })
    count("(c)", await leg("fail_c") { log in try await unsupportedSchemaLeg(model, log) })
    count("(d)", await leg("fail_d") { log in try await cancelStreamLeg(model, log) })

    let passed = counted.filter(\.1).count
    for (name, ok) in counted { #expect(ok, "\(name) failed") }
    say("### SUMMARY: \(passed)/\(counted.count) PASS")
  }

  private var selectedLegs: Set<String>? {
    environment["LITERT_TEST_LEGS"].map { Set($0.split(separator: ",").map(String.init)) }
  }

  // MARK: Counted legs

  private func respondLeg(_ model: LiteRTLMLanguageModel, _ log: LegLog) async throws -> Bool {
    let start = Date()
    let response = try await LanguageModelSession(model: model).respond(to: textPrompt)
    log.line("seconds \(seconds(since: start))")
    log.line("content \(response.content.debugDescription)")
    log.usage(response.usage)
    let nonEmpty = !response.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    let eos = stoppedAtEOS(response.usage)
    log.line("nonEmpty \(nonEmpty) eos \(eos)")
    return nonEmpty && eos
  }

  private func streamLeg(_ model: LiteRTLMLanguageModel, _ log: LegLog) async throws -> Bool {
    let start = Date()
    var snapshots = 0
    var last: LanguageModelSession.ResponseStream<String>.Snapshot?
    for try await snapshot in LanguageModelSession(model: model).streamResponse(to: textPrompt) {
      snapshots += 1
      last = snapshot
    }
    log.line("seconds \(seconds(since: start)) snapshots \(snapshots)")
    guard let last else { return false }
    log.line("content \(last.content.debugDescription)")
    log.usage(last.usage)
    let nonEmpty = !last.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    let eos = stoppedAtEOS(last.usage)
    log.line("nonEmpty \(nonEmpty) eos \(eos)")
    return snapshots >= 1 && nonEmpty && eos
  }

  private func flatLeg(
    _ model: LiteRTLMLanguageModel, _ log: LegLog, includeSchemaInPrompt: Bool = true
  ) async throws -> Bool {
    let start = Date()
    let response = try await LanguageModelSession(model: model).respond(
      to: flatPrompt, generating: CafeReview.self, includeSchemaInPrompt: includeSchemaInPrompt)
    log.line("seconds \(seconds(since: start))")
    log.line("raw \(response.rawContent.jsonString)")
    log.usage(response.usage)
    let violation = violations(response.content)
    log.line("value name:\(response.content.name.debugDescription) rating:\(response.content.rating) price:\(response.content.price) hasWifi:\(response.content.hasWifi) constraint:\(violation ?? "ok")")
    return violation == nil
  }

  private func nestedLeg(
    _ model: LiteRTLMLanguageModel, _ log: LegLog, includeSchemaInPrompt: Bool = true
  ) async throws -> Bool {
    let start = Date()
    let response = try await LanguageModelSession(model: model).respond(
      to: nestedPrompt, generating: TripPlan.self, includeSchemaInPrompt: includeSchemaInPrompt)
    log.line("seconds \(seconds(since: start))")
    log.line("raw \(response.rawContent.jsonString)")
    log.usage(response.usage)
    let violation = violations(response.content)
    log.line("value title:\(response.content.title.debugDescription) stops:\(response.content.stops.count) needsCar:\(response.content.needsCar) constraint:\(violation ?? "ok")")
    return violation == nil
  }

  // MARK: Recorded legs

  private func flatStreamLeg(_ model: LiteRTLMLanguageModel, _ log: LegLog) async throws -> Bool {
    var snapshots = 0
    let stream = LanguageModelSession(model: model).streamResponse(to: flatPrompt, generating: CafeReview.self)
    for try await _ in stream { snapshots += 1 }
    let collected = try await LanguageModelSession(model: model)
      .streamResponse(to: flatPrompt, generating: CafeReview.self).collect()
    log.line("snapshots \(snapshots) (first stream); collected raw \(collected.rawContent.jsonString)")
    let violation = violations(collected.content)
    log.line("collected constraint:\(violation ?? "ok")")
    return violation == nil
  }

  private func nestedStreamLeg(_ model: LiteRTLMLanguageModel, _ log: LegLog) async throws -> Bool {
    var snapshots = 0
    var last: GeneratedContent?
    let stream = LanguageModelSession(model: model).streamResponse(to: nestedPrompt, generating: TripPlan.self)
    for try await snapshot in stream {
      snapshots += 1
      last = snapshot.rawContent
    }
    log.line("snapshots \(snapshots)")
    guard let raw = last else { return false }
    let plan = try TripPlan(raw)
    let violation = violations(plan)
    log.line("last raw \(raw.jsonString) constraint:\(violation ?? "ok")")
    return violation == nil
  }

  private func twoSessionsLeg(_ model: LiteRTLMLanguageModel, _ log: LegLog) async throws -> Bool {
    let start = Date()
    async let first = timed { try await LanguageModelSession(model: model).respond(to: "Name one color.").content }
    async let second = timed { try await LanguageModelSession(model: model).respond(to: "Name one animal.").content }
    let (a, b) = try await (first, second)
    log.line("first \(a.value.debugDescription) done at \(a.end.timeIntervalSince(start)) s")
    log.line("second \(b.value.debugDescription) done at \(b.end.timeIntervalSince(start)) s")
    return !a.value.isEmpty && !b.value.isEmpty
  }

  private func greedyLeg(_ model: LiteRTLMLanguageModel, _ log: LegLog) async throws -> Bool {
    let options = GenerationOptions(sampling: .greedy)
    let first = try await LanguageModelSession(model: model).respond(to: textPrompt, options: options).content
    let second = try await LanguageModelSession(model: model).respond(to: textPrompt, options: options).content
    log.line("first \(first.debugDescription)")
    log.line("second \(second.debugDescription)")
    log.line("same bytes \(first == second)")
    return first == second
  }

  private func maxTokensFlatLeg(_ model: LiteRTLMLanguageModel, _ log: LegLog) async throws -> Bool {
    do {
      let response = try await LanguageModelSession(model: model).respond(
        to: flatPrompt, generating: CafeReview.self,
        options: GenerationOptions(maximumResponseTokens: 8))
      log.line("returned raw \(response.rawContent.jsonString)")
      log.usage(response.usage)
    } catch {
      log.line("threw \(describe(error))")
    }
    return true
  }

  private func maxTokensTextLeg(_ model: LiteRTLMLanguageModel, _ log: LegLog) async throws -> Bool {
    let text = try await LanguageModelSession(model: model).respond(
      to: textPrompt, options: GenerationOptions(maximumResponseTokens: 8))
    log.line("text with 8 tokens \(text.content.debugDescription)")
    log.usage(text.usage)
    return true
  }

  private func nucleusLeg(_ model: LiteRTLMLanguageModel, _ log: LegLog) async throws -> Bool {
    let response = try await LanguageModelSession(model: model).respond(
      to: textPrompt, options: GenerationOptions(sampling: .random(probabilityThreshold: 0.9)))
    log.line("content \(response.content.debugDescription)")
    log.usage(response.usage)
    return !response.content.isEmpty
  }

  private func cancelRespondLeg(_ model: LiteRTLMLanguageModel, _ log: LegLog) async throws -> Bool {
    let task = Task {
      try await LanguageModelSession(model: model).respond(to: "Count from 1 to 500, one number per line.")
    }
    try await Task.sleep(nanoseconds: 3_000_000_000)
    let cancelledAt = Date()
    task.cancel()
    let result = await task.result
    log.line("respond ended \(seconds(since: cancelledAt)) s after cancel: \(outcome(result))")
    return try await nextRequest(model, log)
  }

  // MARK: Failure paths

  private func missingFileLeg(_ model: LiteRTLMLanguageModel, _ scratch: URL, _ log: LegLog) async throws -> Bool {
    let missing = scratch.appendingPathComponent("does-not-exist.litertlm")
    let broken = LiteRTLMLanguageModel(modelFileURL: missing, cacheDirectory: cacheURL)
    log.line("availability \(broken.availability)")
    let missingReported = broken.availability == .unavailable(.modelFileMissing(missing))
    let threw = await throwsOnRespond(broken, log)
    let next = try await nextRequest(model, log)
    return missingReported && threw && next
  }

  private func randomBytesLeg(_ model: LiteRTLMLanguageModel, _ scratch: URL, _ log: LegLog) async throws -> Bool {
    let file = scratch.appendingPathComponent("random.litertlm")
    var bytes = [UInt8](repeating: 0, count: 1 << 20)
    for index in bytes.indices { bytes[index] = UInt8.random(in: 0...255) }
    try Data(bytes).write(to: file)
    return try await brokenFileLeg(model, file, log)
  }

  private func truncatedLeg(_ model: LiteRTLMLanguageModel, _ scratch: URL, _ log: LegLog) async throws -> Bool {
    let file = scratch.appendingPathComponent("truncated.litertlm")
    let source = try FileHandle(forReadingFrom: modelURL)
    defer { try? source.close() }
    try (try source.read(upToCount: 64 << 20) ?? Data()).write(to: file)
    return try await brokenFileLeg(model, file, log)
  }

  private func directoryLeg(_ model: LiteRTLMLanguageModel, _ scratch: URL, _ log: LegLog) async throws -> Bool {
    let directory = scratch.appendingPathComponent("directory.litertlm", isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    return try await brokenFileLeg(model, directory, log)
  }

  private func brokenFileLeg(_ model: LiteRTLMLanguageModel, _ file: URL, _ log: LegLog) async throws -> Bool {
    let broken = LiteRTLMLanguageModel(modelFileURL: file, cacheDirectory: cacheURL)
    log.line("availability before \(broken.availability)")
    let threw = await throwsOnRespond(broken, log)
    let after = broken.availability
    log.line("availability after \(after)")
    var loadFailedReported = false
    if case .unavailable(.loadFailed) = after { loadFailedReported = true }
    let next = try await nextRequest(model, log)
    return threw && loadFailedReported && next
  }

  private func unsupportedSchemaLeg(_ model: LiteRTLMLanguageModel, _ log: LegLog) async throws -> Bool {
    let schema = try JSONDecoder().decode(
      GenerationSchema.self,
      from: Data(##"{"type":"object","properties":{"a":{"$ref":"#/$defs/Missing"}},"required":["a"]}"##.utf8))
    let session = LanguageModelSession(model: model)
    var threw = false
    do {
      let response = try await session.respond(to: "What is 2 + 2?", schema: schema)
      log.line("returned \(response.rawContent.jsonString)")
    } catch {
      threw = !error.localizedDescription.isEmpty
      log.line("threw \(describe(error))")
    }
    do {
      let same = try await session.respond(to: nextPrompt)
      log.line("same session next request: \(same.content.debugDescription)")
    } catch {
      log.line("same session next request threw \(describe(error))")
    }
    let next = try await nextRequest(model, log)
    return threw && next
  }

  private func cancelStreamLeg(_ model: LiteRTLMLanguageModel, _ log: LegLog) async throws -> Bool {
    let counter = SnapshotCounter()
    let task = Task {
      for try await _ in LanguageModelSession(model: model).streamResponse(
        to: "Count from 1 to 500, one number per line.")
      {
        counter.increment()
      }
    }
    let waitStart = Date()
    while counter.value < 5, Date().timeIntervalSince(waitStart) < 120 {
      try await Task.sleep(nanoseconds: 20_000_000)
    }
    let cancelledAt = Date()
    task.cancel()
    let result = await withDeadline(seconds: 60) { await task.result }
    guard let result else {
      log.line("the stream task did not end within 60 s of cancel")
      return false
    }
    log.line(
      "the stream task ended \(seconds(since: cancelledAt)) s after cancel, after \(counter.value) snapshots: \(outcome(result))")
    return try await nextRequest(model, log)
  }

  // MARK: Helpers

  private func throwsOnRespond(_ model: LiteRTLMLanguageModel, _ log: LegLog) async -> Bool {
    do {
      let response = try await LanguageModelSession(model: model).respond(to: nextPrompt)
      log.line("respond returned \(response.content.debugDescription)")
      return false
    } catch {
      log.line("respond threw \(describe(error))")
      return !error.localizedDescription.isEmpty
    }
  }

  private func nextRequest(_ model: LiteRTLMLanguageModel, _ log: LegLog) async throws -> Bool {
    let start = Date()
    let reply = try await LanguageModelSession(model: model).respond(to: nextPrompt).content
    log.line("next request (\(seconds(since: start)) s): \(reply.debugDescription)")
    return !reply.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
  }

  /// The reply ended at the end-of-sequence token: tokens were generated and the KV cache
  /// (prompt and reply) did not reach the 2048-token context.
  private func stoppedAtEOS(_ usage: LanguageModelSession.Usage) -> Bool {
    usage.output.totalTokenCount >= 1 && usage.input.totalTokenCount + usage.output.totalTokenCount < 2048
  }

  private func violations(_ review: CafeReview) -> String? {
    if !(1...5).contains(review.rating) { return "rating \(review.rating) outside 1...5" }
    if !["cheap", "moderate", "expensive"].contains(review.price) { return "price \(review.price) outside the choices" }
    return nil
  }

  private func violations(_ plan: TripPlan) -> String? {
    if !(1...4).contains(plan.stops.count) { return "stops.count \(plan.stops.count) outside 1...4" }
    let kinds: Set<String> = ["temple", "museum", "park", "restaurant", "shop"]
    for (index, stop) in ([plan.start] + plan.stops).enumerated() {
      if !(15...240).contains(stop.minutes) { return "stop \(index) minutes \(stop.minutes) outside 15...240" }
      if !kinds.contains(stop.kind) { return "stop \(index) kind \(stop.kind) outside the choices" }
    }
    return nil
  }

  /// Runs one leg, or returns `nil` when LITERT_TEST_LEGS leaves it out.
  private func leg(_ name: String, limitSeconds: Int = 900, _ body: (LegLog) async throws -> Bool) async -> Bool? {
    if let selectedLegs, !selectedLegs.contains(name) { return nil }
    MemoryGuard.currentStep.withLock { $0 = "leg \(name)" }
    let log = LegLog(name: name, directory: logDirectory)
    let watchdog = DispatchWorkItem {
      print("IT: ### HANG: \(name) after \(limitSeconds) s")
      fflush(stdout)
      exit(3)
    }
    DispatchQueue.global().asyncAfter(deadline: .now() + .seconds(limitSeconds), execute: watchdog)
    defer { watchdog.cancel() }
    let start = Date()
    let passed: Bool
    do {
      passed = try await body(log)
    } catch {
      log.line("FAILED with \(describe(error))")
      passed = false
    }
    let footprint = MemoryGuard.gigabytes(MemoryGuard.footprintBytes())
    log.line("footprint at the end \(footprint)")
    say("\(name): \(passed ? "PASS" : "FAIL") (\(seconds(since: start)) s, footprint \(footprint)) log=\(log.path ?? "-")")
    log.close(passed: passed)
    return passed
  }
}

// MARK: - Support

private func say(_ line: String) {
  print("IT: \(line)")
  fflush(stdout)
}

private func seconds(since start: Date) -> String {
  String(format: "%.1f", Date().timeIntervalSince(start))
}

private func describe(_ error: any Error) -> String {
  switch error {
  case LanguageModelSession.GenerationError.assetsUnavailable(let context):
    return "GenerationError.assetsUnavailable: \(context.debugDescription)"
  case LanguageModelSession.GenerationError.unsupportedGuide(let context):
    return "GenerationError.unsupportedGuide: \(context.debugDescription)"
  case LanguageModelSession.GenerationError.decodingFailure(let context):
    return "GenerationError.decodingFailure: \(context.debugDescription)"
  case LanguageModelSession.GenerationError.exceededContextWindowSize(let context):
    return "GenerationError.exceededContextWindowSize: \(context.debugDescription)"
  case is CancellationError:
    return "CancellationError"
  default:
    return "\(type(of: error)): \(error.localizedDescription)"
  }
}

private func outcome<T>(_ result: Result<T, any Error>) -> String {
  switch result {
  case .success: return "ended without an error"
  case .failure(let error): return "threw \(describe(error))"
  }
}

private func timed<T: Sendable>(_ body: () async throws -> T) async throws -> (value: T, end: Date) {
  let value = try await body()
  return (value, Date())
}

private func withDeadline<T: Sendable>(seconds: Double, _ body: @escaping @Sendable () async -> T) async -> T? {
  await withTaskGroup(of: T?.self) { group in
    group.addTask { await body() }
    group.addTask {
      try? await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
      return nil
    }
    let first = await group.next() ?? nil
    group.cancelAll()
    return first
  }
}

private final class SnapshotCounter: @unchecked Sendable {
  private var count = 0
  private let lock = NSLock()
  func increment() { lock.withLock { count += 1 } }
  var value: Int { lock.withLock { count } }
}

/// One leg's log file.
private final class LegLog: @unchecked Sendable {
  let path: String?
  private var lines: [String] = []
  private let lock = NSLock()

  init(name: String, directory: URL?) {
    path = directory?.appendingPathComponent("\(name).log").path
  }

  func line(_ text: String) {
    lock.withLock { lines.append(text) }
  }

  func usage(_ usage: LanguageModelSession.Usage) {
    let metadata = usage.metadata.sorted { $0.key < $1.key }
      .map { "\($0.key)=\($0.value.jsonString)" }.joined(separator: " ")
    line("usage input \(usage.input.totalTokenCount) output \(usage.output.totalTokenCount) \(metadata)")
  }

  func close(passed: Bool) {
    let text = lock.withLock { (lines + ["RESULT \(passed ? "PASS" : "FAIL")"]).joined(separator: "\n") + "\n" }
    if let path { try? text.write(toFile: path, atomically: true, encoding: .utf8) }
    print(text.split(separator: "\n").map { "IT:   \($0)" }.joined(separator: "\n"))
    fflush(stdout)
  }
}
