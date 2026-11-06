import AnyLanguageModel
import Foundation
import Testing

@testable import AnyLanguageModelLiteRTLM

private func text(_ content: String) -> Transcript.Segment {
  .text(.init(content: content))
}

private func prompt(_ content: String) -> Transcript.Entry {
  .prompt(.init(segments: [text(content)]))
}

private func response(_ content: String) -> Transcript.Entry {
  .response(.init(assetIDs: [], segments: [text(content)]))
}

private func instructions(_ content: String) -> Transcript.Entry {
  .instructions(.init(segments: [text(content)], toolDefinitions: []))
}

private struct WeatherTool: Tool {
  @Generable
  struct Arguments {
    var city: String
  }

  let description = "Looks up the weather."

  func call(arguments: Arguments) async throws -> String { "Sunny" }
}

@Suite struct TranscriptPlanTests {
  @Test func instructionsTwoTurnsAndTheCurrentPromptLast() throws {
    let transcript = Transcript(entries: [
      instructions("Be brief."),
      prompt("Name a color."),
      response("Blue."),
      prompt("Name a fruit."),
      response("Apple."),
      prompt("Which color did you name?"),
    ])
    let plan = try TranscriptPlan(
      transcript: transcript, tools: [], toolCallingMode: nil,
      currentPrompt: Prompt("Which color did you name?"))
    #expect(plan.systemMessage == "Be brief.")
    #expect(
      plan.history == [
        HistoryMessage(role: .user, text: "Name a color."),
        HistoryMessage(role: .model, text: "Blue."),
        HistoryMessage(role: .user, text: "Name a fruit."),
        HistoryMessage(role: .model, text: "Apple."),
      ])
    #expect(plan.prompt == "Which color did you name?")
  }

  @Test func reasoningIsNotSent() throws {
    let transcript = Transcript(entries: [
      prompt("Hi"),
      .reasoning(.init(segments: [text("thinking…")])),
      response("Hello."),
      prompt("Again"),
    ])
    let plan = try TranscriptPlan(
      transcript: transcript, tools: [], toolCallingMode: nil, currentPrompt: Prompt("Again"))
    #expect(
      plan.history == [
        HistoryMessage(role: .user, text: "Hi"), HistoryMessage(role: .model, text: "Hello."),
      ])
    #expect(plan.prompt == "Again")
  }

  @Test func toolEntriesThrow() {
    let call = Transcript.ToolCall(
      id: "c1", toolName: "weather", arguments: GeneratedContent(properties: ["city": "Kyoto"]))
    let calls = Transcript(entries: [prompt("Weather?"), .toolCalls(.init([call])), prompt("And now?")])
    #expect(throws: LiteRTLMLanguageModelError.unsupportedContent("tool calls in the transcript")) {
      try TranscriptPlan(transcript: calls, tools: [], toolCallingMode: nil, currentPrompt: Prompt("And now?"))
    }
    let outputs = Transcript(entries: [
      prompt("Weather?"), .toolOutput(.init(id: "c1", toolName: "weather", segments: [text("Sunny")])),
      prompt("And now?"),
    ])
    #expect(throws: LiteRTLMLanguageModelError.unsupportedContent("tool outputs in the transcript")) {
      try TranscriptPlan(transcript: outputs, tools: [], toolCallingMode: nil, currentPrompt: Prompt("And now?"))
    }
  }

  @Test func imagesThrow() {
    let image = Transcript.Segment.image(.init(data: Data([0xFF, 0xD8]), mimeType: "image/jpeg"))
    let transcript = Transcript(entries: [.prompt(.init(segments: [text("What is this?"), image]))])
    #expect(throws: LiteRTLMLanguageModelError.unsupportedContent("image input")) {
      try TranscriptPlan(
        transcript: transcript, tools: [], toolCallingMode: nil, currentPrompt: Prompt("What is this?"))
    }
  }

  @Test func toolsThrowUnlessDisallowed() throws {
    let transcript = Transcript(entries: [prompt("Hi")])
    #expect(throws: LiteRTLMLanguageModelError.toolsNotSupported) {
      try TranscriptPlan(
        transcript: transcript, tools: [WeatherTool()], toolCallingMode: nil, currentPrompt: Prompt("Hi"))
    }
    let plan = try TranscriptPlan(
      transcript: transcript, tools: [WeatherTool()], toolCallingMode: .disallowed,
      currentPrompt: Prompt("Hi"))
    #expect(plan.prompt == "Hi")
  }

  @Test func withoutATrailingPromptTheWholeTranscriptIsHistory() throws {
    let transcript = Transcript(entries: [prompt("Hi"), response("Hello.")])
    let plan = try TranscriptPlan(
      transcript: transcript, tools: [], toolCallingMode: nil, currentPrompt: Prompt("Next"))
    #expect(plan.history.count == 2)
    #expect(plan.prompt == "Next")
  }

  @Test func structuredSegmentsAreSentAsJSON() throws {
    let content = GeneratedContent(properties: ["rating": 4])
    let transcript = Transcript(entries: [
      .response(.init(assetIDs: [], segments: [.structure(.init(source: "Review", content: content))])),
      prompt("More"),
    ])
    let plan = try TranscriptPlan(
      transcript: transcript, tools: [], toolCallingMode: nil, currentPrompt: Prompt("More"))
    #expect(plan.history == [HistoryMessage(role: .model, text: #"{"rating":4}"#)])
  }

  @Test func aSessionSendsItsTranscript() async throws {
    let setup = try FakeSetup([.reply("Blue.")])
    let session = LanguageModelSession(model: setup.model, instructions: "Be brief.")
    _ = try await session.respond(to: "Name a color.")
    _ = try await session.respond(to: "Which color did you name?")

    let setups = setup.engine.setups.current
    #expect(setups.count == 2)
    #expect(setups[0].systemMessage == "Be brief.")
    #expect(setups[0].history.isEmpty)
    #expect(setups[1].systemMessage == "Be brief.")
    #expect(
      setups[1].history == [
        HistoryMessage(role: .user, text: "Name a color."),
        HistoryMessage(role: .model, text: "Blue."),
      ])
    #expect(setup.engine.lastSent?.text == "Which color did you name?")
    #expect(setups.allSatisfy { !$0.constrained })
  }

  @Test func aSessionWithToolsThrowsBeforeLoading() async throws {
    let setup = try FakeSetup()
    let session = LanguageModelSession(model: setup.model, tools: [WeatherTool()])
    await #expect(throws: LiteRTLMLanguageModelError.toolsNotSupported) {
      _ = try await session.respond(to: "Weather in Kyoto?")
    }
    #expect(setup.loader.loadCount.current == 0)
  }
}
