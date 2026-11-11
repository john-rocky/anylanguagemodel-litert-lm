import AnyLanguageModel
import AnyLanguageModelLiteRTLM
import Foundation
import Testing

// Every Swift block of README.md except the Package.swift lines, compiled here (not run).
// `readmeBlocksMatch` fails when a line of a README block is not found here (in the block's
// order), so a change to one has to be made in both.

enum ReadmeExamples {
  @Generable(description: "A stop on a trip")
  struct Stop {
    @Guide(description: "Name of the place")
    var place: String
    @Guide(description: "Minutes to spend there", .range(15...240))
    var minutes: Int
    @Guide(description: "Kind of place", .anyOf(["temple", "museum", "park", "restaurant", "shop"]))
    var kind: String
  }

  @Generable(description: "A one-day trip plan")
  struct TripPlan {
    @Guide(description: "Title of the plan")
    var title: String
    @Guide(description: "The first stop of the day")
    var start: Stop
    @Guide(description: "The other stops, in order", .count(1...4))
    var stops: [Stop]
    @Guide(description: "Whether the plan needs a car")
    var needsCar: Bool
  }

  static func quickstart() async throws -> LanguageModelSession {
    let model = LiteRTLMLanguageModel(.gemma4_E2B)
    try await model.load()  // downloads about 2.6 GB on the first run
    let session = LanguageModelSession(model: model)
    let response = try await session.respond(to: "Write a haiku about the sea.")
    print(response.content)
    return session
  }

  static func structuredOutput(_ session: LanguageModelSession) async throws {
    let plan = try await session.respond(
      to: "Plan a one-day trip in Kyoto.",
      generating: TripPlan.self)
    print(plan.content.title, plan.content.stops.map(\.place))
  }

  static func streaming(_ session: LanguageModelSession) {
    let task = Task {
      for try await snapshot in session.streamResponse(to: "Tell me about Kyoto.") {
        print(snapshot.content)  // the text so far
      }
    }
    // To stop, cancel the task. The backend stops the generation in LiteRT-LM.
    task.cancel()
  }

  static func structuredStreaming(_ session: LanguageModelSession) async throws {
    for try await snapshot in session.streamResponse(
      to: "Plan a one-day trip in Kyoto.",
      generating: TripPlan.self
    ) {
      let partial = snapshot.content  // TripPlan.PartiallyGenerated
      print(partial.title ?? "…", partial.stops?.count ?? 0)
    }
  }

  static func ownModelFile(_ fileURL: URL) -> [LiteRTLMLanguageModel] {
    // A .litertlm on disk: bundled, sideloaded or fine-tuned.
    let local = LiteRTLMLanguageModel(modelFileURL: fileURL)

    // Any public .litertlm on Hugging Face, downloaded on first use.
    let hub = LiteRTLMLanguageModel(
      huggingFaceRepo: "litert-community/gemma-4-E2B-it-litert-lm",
      fileName: "gemma-4-E2B-it.litertlm")
    return [local, hub]
  }

  static func availability(_ model: LiteRTLMLanguageModel) {
    switch model.availability {
    case .available:
      break
    case .unavailable(.notDownloaded):
      print("The model downloads on the first request.")
    case .unavailable(let reason):
      print("Not available: \(reason)")
    }
  }
}

@Suite struct ReadmeExamplesTests {
  @Test func readmeBlocksMatch() throws {
    let tests = URL(fileURLWithPath: #filePath)
    let root = tests.deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    let readme = try String(contentsOf: root.appendingPathComponent("README.md"), encoding: .utf8)
    let here = try String(contentsOf: tests, encoding: .utf8)
    let hereLines = lines(here)
    let blocks = swiftBlocks(readme).filter { !$0.contains("Package.swift") }
    #expect(blocks.count >= 6)
    for block in blocks {
      // Each line of the block, in order (the file splits blocks into functions).
      var next = hereLines.startIndex
      var missing: String?
      for line in lines(block) where !line.hasPrefix("import ") {
        guard let found = hereLines[next...].firstIndex(of: line) else {
          missing = line
          break
        }
        next = found + 1
      }
      #expect(missing == nil, "README line not in ReadmeExamples.swift, in order: \(missing ?? "")")
    }
    for line in Set(blocks.flatMap(lines).filter { $0.hasPrefix("import ") }) {
      #expect(hereLines.contains(line), "README import not in ReadmeExamples.swift: \(line)")
    }
  }

  /// The non-empty lines, trimmed.
  private func lines(_ text: String) -> [String] {
    text.split(separator: "\n").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
  }

  private func swiftBlocks(_ markdown: String) -> [String] {
    var blocks: [String] = []
    var current: [Substring]?
    for line in markdown.split(separator: "\n", omittingEmptySubsequences: false) {
      if current == nil, line.hasPrefix("```swift") {
        current = []
      } else if current != nil, line.hasPrefix("```") {
        blocks.append(current!.joined(separator: "\n"))
        current = nil
      } else {
        current?.append(line)
      }
    }
    return blocks
  }
}
