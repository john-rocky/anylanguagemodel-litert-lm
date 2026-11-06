// LiteRTLMDemo self-test: the package's checks on a device, without the UI.
//
// Launch with LITERT_SELFTEST=all, or a comma list of respond, stream, flat, nested and
// failures. LITERT_MODEL_PATH points at a model file already on disk (otherwise the app
// downloads Gemma 4 E2B first). LITERT_SELFTEST_RUNS sets how many flat and nested runs
// (default 3). LITERT_CACHE_DIR sets where LiteRT-LM writes its GPU cache (default: the app's
// Caches). LITERT_SELFTEST_MEMORY_LIMIT_GB stops the run when the app's memory footprint
// passes it (default 4 on iOS, 10 on macOS).
//
// Every line starts with "ST:" and also shows on the screen. The run ends with
// `### DEVICE: <machine>`, `### SELFTEST: <passed>/<run> PASS` and `ST: DONE`; a step that
// takes longer than 900 s prints `### HANG` and exits.
//
// Pass rules: respond and stream end at the end-of-sequence token with non-empty text
// (fewer than the 2048-token context, at least one token); flat and nested return the type
// within its @Guide ranges; each failure path throws an error with a description and a new
// session on the real model answers afterwards; the cancelled stream's task ends within 60 s.

import AnyLanguageModel
import AnyLanguageModelLiteRTLM
import Darwin
import Foundation
import Observation
import SwiftUI
import os

enum SelfTest {
  private static let environment = ProcessInfo.processInfo.environment
  private static let logger = Logger(subsystem: "com.example.litertlmdemo", category: "SelfTest")
  private static let hangLimitSeconds = 900
  private static let contextTokens = 2048

  static var isRequested: Bool { environment["LITERT_SELFTEST"] != nil }

  private static var selected: Set<String> {
    let value = environment["LITERT_SELFTEST"] ?? ""
    if value == "all" || value == "1" { return ["respond", "stream", "flat", "nested", "failures"] }
    return Set(value.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) })
  }

  private static var runs: Int { max(1, Int(environment["LITERT_SELFTEST_RUNS"] ?? "") ?? 3) }

  private static var cacheDirectory: URL? {
    environment["LITERT_CACHE_DIR"].map { URL(fileURLWithPath: $0, isDirectory: true) }
  }

  static func run() async {
    startMemoryGuard()
    log("start — device=\(deviceName()) OS=\(ProcessInfo.processInfo.operatingSystemVersionString)")
    let model: LiteRTLMLanguageModel
    let modelFile: URL
    if let path = environment["LITERT_MODEL_PATH"] {
      modelFile = URL(fileURLWithPath: path)
      model = LiteRTLMLanguageModel(
        modelFileURL: modelFile, cacheDirectory: cacheDirectory, enableBenchmark: true)
    } else {
      modelFile = catalogFile()
      model = LiteRTLMLanguageModel(
        .gemma4_E2B, cacheDirectory: cacheDirectory, enableBenchmark: true)
    }
    log("model \(modelFile.path)")
    do {
      try await guarded("load") { try await model.load() }
      log("loaded")
    } catch {
      log("FATAL could not load the model: \(quoted(error))")
      return
    }

    var outcomes: [Bool] = []
    let legs = selected
    if legs.contains("respond") { outcomes.append(await guarded("respond") { await respond(model) }) }
    if legs.contains("stream") { outcomes.append(await guarded("stream") { await stream(model) }) }
    if legs.contains("flat") {
      for run in 1...runs {
        outcomes.append(await guarded("flat \(run)") { await flat(model, run: run) })
      }
    }
    if legs.contains("nested") {
      for run in 1...runs {
        outcomes.append(await guarded("nested \(run)") { await nested(model, run: run) })
      }
    }
    if legs.contains("failures") {
      let scratch = FileManager.default.temporaryDirectory
        .appendingPathComponent("litertlm-selftest-\(UUID().uuidString)", isDirectory: true)
      defer { try? FileManager.default.removeItem(at: scratch) }
      do {
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
        outcomes.append(await guarded("(a)") { await missingFile(model, scratch: scratch) })
        outcomes.append(await guarded("(b1)") { await randomBytes(model, scratch: scratch) })
        outcomes.append(
          await guarded("(b2)") { await truncated(model, scratch: scratch, source: modelFile) })
        outcomes.append(await guarded("(b3)") { await directory(model, scratch: scratch) })
        outcomes.append(await guarded("(c)") { await unsupportedSchema(model) })
        outcomes.append(await guarded("(d)") { await cancelStream(model) })
      } catch {
        log("FATAL could not prepare the failure-path files: \(error.localizedDescription)")
      }
    }
    emit("### DEVICE: \(deviceName())")
    emit("### SELFTEST: \(outcomes.filter { $0 }.count)/\(outcomes.count) PASS")
    log("DONE")
  }

  // MARK: Legs

  private static let textPrompt = "Write a haiku about the sea."
  private static let nextPrompt = "Reply with one short sentence about the sea."

  private static func respond(_ model: LiteRTLMLanguageModel) async -> Bool {
    do {
      let start = Date()
      let response = try await LanguageModelSession(model: model).respond(to: textPrompt)
      let text = response.content
      let eos = stoppedAtEOS(response.usage)
      let passed = eos && !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
      log(
        "respond: \(verdict(passed)) · eos=\(eos) chars=\(text.count) tokens=\(response.usage.output.totalTokenCount) "
          + "decode=\(number(response.usage, "decodeTokensPerSecond")) tok/s "
          + "seconds=\(seconds(since: start)) · \(oneLine(text))")
      return passed
    } catch {
      log("respond: FAIL · threw \(quoted(error))")
      return false
    }
  }

  private static func stream(_ model: LiteRTLMLanguageModel) async -> Bool {
    do {
      var chunks = 0
      var last: LanguageModelSession.ResponseStream<String>.Snapshot?
      for try await snapshot in LanguageModelSession(model: model).streamResponse(to: textPrompt) {
        chunks += 1
        last = snapshot
      }
      guard let last else {
        log("stream: FAIL · no snapshot")
        return false
      }
      let eos = stoppedAtEOS(last.usage)
      let passed = eos && !last.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
      log(
        "stream: \(verdict(passed)) · chunks=\(chunks) eos=\(eos) "
          + "decode=\(number(last.usage, "decodeTokensPerSecond")) tok/s · \(oneLine(last.content))")
      return passed
    } catch {
      log("stream: FAIL · threw \(quoted(error))")
      return false
    }
  }

  private static func flat(_ model: LiteRTLMLanguageModel, run: Int) async -> Bool {
    do {
      let start = Date()
      let response = try await LanguageModelSession(model: model).respond(
        to: "Invent a small cafe and review it.", generating: CafeReview.self)
      let review = response.content
      let problem = violation(review)
      log(
        "flat \(run)/\(runs): \(verdict(problem == nil)) · \(seconds(since: start)) s · "
          + "\(response.rawContent.jsonString)\(problem.map { " · \($0)" } ?? "")")
      return problem == nil
    } catch {
      log("flat \(run)/\(runs): FAIL · threw \(quoted(error))")
      return false
    }
  }

  private static func nested(_ model: LiteRTLMLanguageModel, run: Int) async -> Bool {
    do {
      let start = Date()
      let response = try await LanguageModelSession(model: model).respond(
        to: "Plan a one-day trip in Kyoto.", generating: TripPlan.self)
      let plan = response.content
      let problem = violation(plan)
      log(
        "nested \(run)/\(runs): \(verdict(problem == nil)) · \(seconds(since: start)) s · "
          + "stops=\(plan.stops.count) · \(response.rawContent.jsonString)\(problem.map { " · \($0)" } ?? "")")
      return problem == nil
    } catch {
      log("nested \(run)/\(runs): FAIL · threw \(quoted(error))")
      return false
    }
  }

  // MARK: Failure paths

  private static func missingFile(_ model: LiteRTLMLanguageModel, scratch: URL) async -> Bool {
    let missing = scratch.appendingPathComponent("does-not-exist.litertlm")
    let broken = LiteRTLMLanguageModel(modelFileURL: missing, cacheDirectory: cacheDirectory)
    let reported = broken.availability == .unavailable(.modelFileMissing(missing))
    return await failurePath("(a) model file missing", broken: broken, real: model, extraCheck: reported)
  }

  private static func randomBytes(_ model: LiteRTLMLanguageModel, scratch: URL) async -> Bool {
    let file = scratch.appendingPathComponent("random.litertlm")
    var bytes = [UInt8](repeating: 0, count: 1 << 20)
    for index in bytes.indices { bytes[index] = UInt8.random(in: 0...255) }
    do { try Data(bytes).write(to: file) } catch {
      log("(b1) random bytes: FAIL · could not write the file: \(error.localizedDescription)")
      return false
    }
    return await brokenFile("(b1) random bytes", file: file, real: model)
  }

  private static func truncated(_ model: LiteRTLMLanguageModel, scratch: URL, source: URL) async -> Bool {
    let file = scratch.appendingPathComponent("truncated.litertlm")
    do {
      let handle = try FileHandle(forReadingFrom: source)
      defer { try? handle.close() }
      try (try handle.read(upToCount: 64 << 20) ?? Data()).write(to: file)
    } catch {
      log("(b2) first 64 MB of the real model: FAIL · could not write the file: \(error.localizedDescription)")
      return false
    }
    return await brokenFile("(b2) first 64 MB of the real model", file: file, real: model)
  }

  private static func directory(_ model: LiteRTLMLanguageModel, scratch: URL) async -> Bool {
    let folder = scratch.appendingPathComponent("directory.litertlm", isDirectory: true)
    do { try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true) } catch {
      log("(b3) a directory: FAIL · could not create it: \(error.localizedDescription)")
      return false
    }
    return await brokenFile("(b3) a directory", file: folder, real: model)
  }

  private static func brokenFile(_ name: String, file: URL, real: LiteRTLMLanguageModel) async -> Bool {
    let broken = LiteRTLMLanguageModel(modelFileURL: file, cacheDirectory: cacheDirectory)
    return await failurePath(name, broken: broken, real: real) {
      if case .unavailable(.loadFailed) = broken.availability { return true }
      return false
    }
  }

  /// Runs a request on `broken`, which must throw, then one on `real`, which must answer.
  private static func failurePath(
    _ name: String, broken: LiteRTLMLanguageModel, real: LiteRTLMLanguageModel,
    extraCheck: Bool = true, after: () -> Bool = { true }
  ) async -> Bool {
    let thrown: String
    do {
      let response = try await LanguageModelSession(model: broken).respond(to: nextPrompt)
      log("\(name): FAIL · returned \(oneLine(response.content))")
      return false
    } catch {
      guard !error.localizedDescription.isEmpty else {
        log("\(name): FAIL · threw an error without a description")
        return false
      }
      thrown = quoted(error)
    }
    let reportedAfter = after()
    let next = await nextRequest(real)
    let passed = extraCheck && reportedAfter && next != nil
    log(
      "\(name): \(verdict(passed)) · threw \(thrown); availability \(extraCheck && reportedAfter ? "ok" : "wrong"); "
        + "next request: \(next.map(oneLine) ?? "failed")")
    return passed
  }

  private static func unsupportedSchema(_ model: LiteRTLMLanguageModel) async -> Bool {
    let name = "(c) unsupported schema"
    let schema: GenerationSchema
    do {
      schema = try JSONDecoder().decode(
        GenerationSchema.self,
        from: Data(##"{"type":"object","properties":{"a":{"$ref":"#/$defs/Missing"}},"required":["a"]}"##.utf8))
    } catch {
      log("\(name): FAIL · could not build the schema: \(error)")
      return false
    }
    let thrown: String
    do {
      let response = try await LanguageModelSession(model: model).respond(to: "What is 2 + 2?", schema: schema)
      log("\(name): FAIL · returned \(response.rawContent.jsonString)")
      return false
    } catch {
      guard !error.localizedDescription.isEmpty else {
        log("\(name): FAIL · threw an error without a description")
        return false
      }
      thrown = quoted(error)
    }
    let next = await nextRequest(model)
    log("\(name): \(verdict(next != nil)) · threw \(thrown); next request: \(next.map(oneLine) ?? "failed")")
    return next != nil
  }

  private static func cancelStream(_ model: LiteRTLMLanguageModel) async -> Bool {
    let name = "(d) cancel during a stream"
    let counter = Counter()
    let task = Task {
      for try await _ in LanguageModelSession(model: model).streamResponse(
        to: "Count from 1 to 500, one number per line.")
      {
        counter.increment()
      }
    }
    let waitStart = Date()
    while counter.value < 5, Date().timeIntervalSince(waitStart) < 120 {
      try? await Task.sleep(nanoseconds: 5_000_000)
    }
    let snapshots = counter.value
    let cancelledAt = Date()
    task.cancel()
    let ended = await withDeadline(seconds: 60) { await task.result }
    guard let ended else {
      log("\(name): FAIL · the stream's task did not end within 60 s of cancel()")
      return false
    }
    let how: String
    switch ended {
    case .success: how = "the loop ended"
    case .failure(let error): how = "threw \(error is CancellationError ? "CancellationError" : quoted(error))"
    }
    let next = await nextRequest(model)
    log(
      "\(name): \(verdict(next != nil)) · \(how) \(seconds(since: cancelledAt)) s after cancel(), "
        + "cancelled at snapshot \(snapshots); next request: \(next.map(oneLine) ?? "failed")")
    return next != nil
  }

  // MARK: Helpers

  private static func nextRequest(_ model: LiteRTLMLanguageModel) async -> String? {
    guard let reply = try? await LanguageModelSession(model: model).respond(to: nextPrompt).content,
      !reply.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    else { return nil }
    return reply
  }

  private static func stoppedAtEOS(_ usage: LanguageModelSession.Usage) -> Bool {
    usage.output.totalTokenCount >= 1
      && usage.input.totalTokenCount + usage.output.totalTokenCount < contextTokens
  }

  private static func violation(_ review: CafeReview) -> String? {
    if !(1...5).contains(review.rating) { return "rating \(review.rating) outside 1...5" }
    if !["cheap", "moderate", "expensive"].contains(review.price) { return "price \(review.price) outside the choices" }
    return nil
  }

  private static func violation(_ plan: TripPlan) -> String? {
    if !(1...4).contains(plan.stops.count) { return "stops.count \(plan.stops.count) outside 1...4" }
    let kinds: Set<String> = ["temple", "museum", "park", "restaurant", "shop"]
    for (index, stop) in ([plan.start] + plan.stops).enumerated() {
      if !(15...240).contains(stop.minutes) { return "stop \(index) minutes \(stop.minutes) outside 15...240" }
      if !kinds.contains(stop.kind) { return "stop \(index) kind \(stop.kind) outside the choices" }
    }
    return nil
  }

  private static func number(_ usage: LanguageModelSession.Usage, _ key: String) -> String {
    guard let value = usage.metadata[key], let number = try? Double(value) else { return "n/a" }
    return String(format: "%.1f", number)
  }

  /// Where `LiteRTLMLanguageModel(.gemma4_E2B)` keeps its download by default.
  private static func catalogFile() -> URL {
    FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
      .appendingPathComponent("LiteRTModels", isDirectory: true)
      .appendingPathComponent("gemma-4-E2B-it.litertlm")
  }

  private static func verdict(_ passed: Bool) -> String { passed ? "PASS" : "FAIL" }

  private static func seconds(since start: Date) -> String {
    String(format: "%.1f", Date().timeIntervalSince(start))
  }

  private static func oneLine(_ text: String) -> String {
    String(text.replacingOccurrences(of: "\n", with: " ").trimmingCharacters(in: .whitespaces).prefix(240))
  }

  private static func quoted(_ error: any Error) -> String {
    let text: String
    switch error {
    case LanguageModelSession.GenerationError.assetsUnavailable(let context),
      LanguageModelSession.GenerationError.unsupportedGuide(let context),
      LanguageModelSession.GenerationError.decodingFailure(let context),
      LanguageModelSession.GenerationError.exceededContextWindowSize(let context):
      text = "\(type(of: error)).\(caseName(error)): \(context.debugDescription)"
    default:
      text = error.localizedDescription
    }
    return "\"\(oneLine(text))\""
  }

  private static func caseName(_ error: any Error) -> String {
    String(describing: error).split(separator: "(").first.map(String.init) ?? "error"
  }

  /// Runs one step under the hang watchdog.
  private static func guarded<T>(_ step: String, _ body: () async throws -> T) async rethrows -> T {
    currentStep.withLock { $0 = step }
    let watchdog = DispatchWorkItem {
      emit("### HANG: (\(step)) after \(hangLimitSeconds) s")
      exit(3)
    }
    DispatchQueue.global().asyncAfter(deadline: .now() + .seconds(hangLimitSeconds), execute: watchdog)
    defer { watchdog.cancel() }
    return try await body()
  }

  private static let currentStep = OSAllocatedUnfairLock(initialState: "start")

  /// Stops the run before a runaway allocation takes the device's memory.
  private static func startMemoryGuard() {
    #if os(iOS)
      let fallback = 4.0
    #else
      let fallback = 10.0
    #endif
    let limitGB = Double(environment["LITERT_SELFTEST_MEMORY_LIMIT_GB"] ?? "") ?? fallback
    let limit = UInt64(limitGB * 1_073_741_824)
    let thread = Thread {
      while true {
        let footprint = footprintBytes()
        if footprint > limit {
          emit(
            "ST: FATAL memory footprint \(String(format: "%.2f", Double(footprint) / 1_073_741_824)) GB "
              + "> \(limitGB) GB during \(currentStep.withLock { $0 })")
          exit(4)
        }
        Thread.sleep(forTimeInterval: 0.1)
      }
    }
    thread.qualityOfService = .userInteractive
    thread.start()
  }

  private static func footprintBytes() -> UInt64 {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(
      MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
    let result = withUnsafeMutablePointer(to: &info) { pointer in
      pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
        task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
      }
    }
    return result == KERN_SUCCESS ? info.phys_footprint : 0
  }

  private static func log(_ message: String) { emit("ST: \(message)") }

  /// One line on stdout, in the unified log, and on the screen.
  private static func emit(_ line: String) {
    logger.log("\(line, privacy: .public)")
    print(line)
    fflush(stdout)
    Task { @MainActor in SelfTestOutput.shared.lines.append(line) }
  }

  private static func deviceName() -> String {
    #if os(macOS)
      var size = 0
      sysctlbyname("hw.model", nil, &size, nil, 0)
      var name = [CChar](repeating: 0, count: size)
      sysctlbyname("hw.model", &name, &size, nil, 0)
      return String(decoding: name.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
    #else
      var info = utsname()
      uname(&info)
      return withUnsafeBytes(of: &info.machine) { raw in
        String(cString: raw.baseAddress!.assumingMemoryBound(to: CChar.self))
      }
    #endif
  }
}

private final class Counter: Sendable {
  private let count = OSAllocatedUnfairLock(initialState: 0)
  func increment() { count.withLock { $0 += 1 } }
  var value: Int { count.withLock { $0 } }
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

@MainActor @Observable
final class SelfTestOutput {
  static let shared = SelfTestOutput()
  var lines: [String] = []
}

/// The screen while the self-test runs; the results also go to stdout.
struct SelfTestView: View {
  @State private var output = SelfTestOutput.shared
  @State private var started = false

  var body: some View {
    ScrollView {
      VStack(alignment: .leading, spacing: 4) {
        Text("LiteRTLMDemo self-test").font(.headline)
        ForEach(Array(output.lines.enumerated()), id: \.offset) { _, line in
          Text(line).font(.caption.monospaced()).textSelection(.enabled)
        }
      }
      .padding()
      .frame(maxWidth: .infinity, alignment: .leading)
    }
    .task {
      guard !started else { return }
      started = true
      await Task.detached { await SelfTest.run() }.value
    }
  }
}
