import AnyLanguageModel
import Foundation
import Testing

@testable import AnyLanguageModelLiteRTLM

@Suite struct StructuredOutputTests {
  @Test func sortedKeysGiveTheSameSchemaEveryTime() throws {
    let first = try StructuredOutput.schemaJSON(for: TripPlan.generationSchema)
    let second = try StructuredOutput.schemaJSON(for: TripPlan.generationSchema)
    let third = try StructuredOutput.schemaJSON(for: TripPlan.generationSchema)
    #expect(first == second)
    #expect(second == third)
    #expect(first.contains(#""$defs""#))
  }

  @Test func theKeyLineListsTheRootKeysAlphabetically() throws {
    let cafe = try StructuredOutput.schemaJSON(for: CafeReview.generationSchema)
    #expect(StructuredOutput.keyLine(forSchemaJSON: cafe) == "Reply in JSON with the keys hasWifi, name, price and rating.")
    let trip = try StructuredOutput.schemaJSON(for: TripPlan.generationSchema)
    #expect(StructuredOutput.keyLine(forSchemaJSON: trip) == "Reply in JSON with the keys needsCar, start, stops and title.")
    #expect(
      StructuredOutput.keyLine(forSchemaJSON: #"{"type":"object","properties":{"answer":{"type":"string"}}}"#)
        == "Reply in JSON with the key answer.")
    #expect(StructuredOutput.keyLine(forSchemaJSON: #"{"type":"string"}"#) == "Reply in JSON.")
    #expect(StructuredOutput.keyLine(forSchemaJSON: #"{"type":"object","properties":{"b":{},"a":{}}}"#) == "Reply in JSON with the keys a and b.")
  }

  @Test func aStringRequestIsNotConstrained() async throws {
    let setup = try FakeSetup([.reply("A haiku.")])
    let session = LanguageModelSession(model: setup.model)
    let response = try await session.respond(to: "Write a haiku.")
    #expect(response.content == "A haiku.")
    #expect(setup.engine.lastSent?.schemaJSON == nil)
    #expect(setup.engine.lastSent?.text == "Write a haiku.")
    #expect(setup.engine.lastSetup?.constrained == false)
  }

  @Test func aGenerableRequestSendsTheSchemaAndTheKeyLine() async throws {
    let setup = try FakeSetup([.reply(cafeJSON)])
    let session = LanguageModelSession(model: setup.model)
    let response = try await session.respond(to: "Invent a cafe.", generating: CafeReview.self)
    #expect(response.content.name == "The Daily Grind")
    #expect(response.content.rating == 4)
    let sent = try #require(setup.engine.lastSent)
    #expect(sent.text == "Invent a cafe.\n\nReply in JSON with the keys hasWifi, name, price and rating.")
    #expect(sent.schemaJSON == (try StructuredOutput.schemaJSON(for: CafeReview.generationSchema)))
    #expect(setup.engine.lastSetup?.constrained == true)
  }

  @Test func includeSchemaInPromptFalseLeavesThePromptAlone() async throws {
    let setup = try FakeSetup([.reply(cafeJSON)])
    let session = LanguageModelSession(model: setup.model)
    _ = try await session.respond(to: "Invent a cafe.", generating: CafeReview.self, includeSchemaInPrompt: false)
    #expect(setup.engine.lastSent?.text == "Invent a cafe.")
    #expect(setup.engine.lastSent?.schemaJSON != nil)
  }

  @Test func theSchemaFormUsesTheGivenSchema() async throws {
    let setup = try FakeSetup([.reply(#"{"answer":"zebra-7"}"#)])
    let session = LanguageModelSession(model: setup.model)
    let schema = try JSONDecoder().decode(
      GenerationSchema.self,
      from: Data(#"{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}"#.utf8))
    let response = try await session.respond(to: "Answer.", schema: schema)
    #expect(response.content.jsonString == #"{"answer":"zebra-7"}"#)
    #expect(setup.engine.lastSent?.schemaJSON?.contains(#""answer""#) == true)
  }

  @Test func aCutOffReplyIsADecodingFailure() {
    #expect(throws: LanguageModelSession.GenerationError.self) {
      _ = try StructuredOutput.decode(#"{"hasWifi":true,"name":"The Da"#, as: CafeReview.self)
    }
    do {
      _ = try StructuredOutput.decode(#"{"hasWifi":true"#, as: CafeReview.self)
      Issue.record("decoded a cut-off reply")
    } catch LanguageModelSession.GenerationError.decodingFailure(let context) {
      #expect(context.debugDescription.contains("not complete JSON"))
    } catch {
      Issue.record("unexpected error \(error)")
    }
  }

  @Test func aReplyOfTheWrongShapeIsADecodingFailure() {
    do {
      _ = try StructuredOutput.decode(#"{"title":"x"}"#, as: CafeReview.self)
      Issue.record("decoded the wrong shape")
    } catch LanguageModelSession.GenerationError.decodingFailure(let context) {
      #expect(context.debugDescription.contains("did not decode as CafeReview"))
    } catch {
      Issue.record("unexpected error \(error)")
    }
  }

  @Test func aStructuredStreamYieldsPartialsThenTheWholeValue() async throws {
    let deltas = [#"{"hasWifi":true,"#, #""name":"The Daily"#, #" Grind","price":"moderate","#, #""rating":4}"#]
    let setup = try FakeSetup([.deltas(deltas)])
    let session = LanguageModelSession(model: setup.model)
    var snapshots: [CafeReview.PartiallyGenerated] = []
    for try await snapshot in session.streamResponse(to: "Invent a cafe.", generating: CafeReview.self) {
      snapshots.append(snapshot.content)
    }
    #expect(snapshots.count >= 2)
    #expect(snapshots.last?.name == "The Daily Grind")
    #expect(snapshots.last?.rating == 4)
    #expect(setup.engine.lastSent?.streamed == true)
  }

  @Test func aTextStreamYieldsTheTextSoFar() async throws {
    let setup = try FakeSetup([.deltas(["The ", "sea ", "sleeps."])])
    let session = LanguageModelSession(model: setup.model)
    var texts: [String] = []
    for try await snapshot in session.streamResponse(to: "Write a haiku.") {
      texts.append(snapshot.content)
    }
    #expect(texts.prefix(3) == ["The ", "The sea ", "The sea sleeps."])
    #expect(texts.last == "The sea sleeps.")
    let collected = try await LanguageModelSession(model: setup.model).streamResponse(to: "Again").collect()
    #expect(collected.content == "The sea sleeps.")
  }

  @Test func benchmarkNumbersBecomeUsage() async throws {
    let numbers = RuntimeBenchmark(
      prefillTokens: 12, decodeTokens: 30, prefillTokensPerSecond: 400, decodeTokensPerSecond: 80.5,
      timeToFirstTokenSeconds: 0.2, kvCacheTokens: 50)
    let setup = try FakeSetup([.reply("Hi")], benchmark: numbers)
    let response = try await LanguageModelSession(model: setup.model).respond(to: "Hello")
    #expect(response.usage.input.totalTokenCount == 20)
    #expect(response.usage.output.totalTokenCount == 30)
    #expect(response.usage.metadata["decodeTokensPerSecond"] == GeneratedContent(80.5))
    let noKV = LiteRTLMLanguageModel.usage(
      RuntimeBenchmark(
        prefillTokens: 12, decodeTokens: 30, prefillTokensPerSecond: 0, decodeTokensPerSecond: 0,
        timeToFirstTokenSeconds: 0, kvCacheTokens: nil))
    #expect(noKV.input.totalTokenCount == 12)
    #expect(LiteRTLMLanguageModel.usage(nil).totalTokenCount == 0)
  }
}
