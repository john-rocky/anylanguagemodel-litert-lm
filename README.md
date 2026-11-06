# AnyLanguageModel LiteRT-LM

Run Gemma 4 and other `.litertlm` models on iPhone and Mac through
[AnyLanguageModel](https://github.com/huggingface/AnyLanguageModel)'s `LanguageModelSession`.
`@Generable` output uses [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM)'s constrained decoding.
<!-- r3: runtime + one device + one number + date, e.g.
On an iPhone 18 Pro (iOS 27.2) with LiteRT-LM v0.19.0, Gemma 4 E2B … (measured 2026-11-..). -->

Community package, not affiliated with Google, Apple or Hugging Face.

On iOS 27 and macOS 27 there is another way to reach LiteRT-LM from AnyLanguageModel:
`FoundationLanguageModel` can wrap the Foundation Models backend of
[swift-litert-lm](https://github.com/john-rocky/swift-litert-lm) (`LiteRTLanguageModel`), which steers
structured output with the prompt only and uses a fixed sampler. This package conforms to
AnyLanguageModel's `LanguageModel` directly: it builds for iOS 17 and macOS 14 and later, enforces the
`@Generable` schema with constrained decoding (llguidance), and applies `GenerationOptions`.

## Quickstart

```swift
// Package.swift
.package(url: "https://github.com/huggingface/AnyLanguageModel", from: "1.2.0"),
.package(url: "https://github.com/john-rocky/anylanguagemodel-litert-lm", from: "0.1.0"),

// target dependencies
.product(name: "AnyLanguageModel", package: "AnyLanguageModel"),
.product(name: "AnyLanguageModelLiteRTLM", package: "anylanguagemodel-litert-lm"),
```

```swift
import AnyLanguageModel
import AnyLanguageModelLiteRTLM

let model = LiteRTLMLanguageModel(.gemma4_E2B)
try await model.load()  // downloads about 2.6 GB on the first run
let session = LanguageModelSession(model: model)
let response = try await session.respond(to: "Write a haiku about the sea.")
print(response.content)
```

`load()` downloads the model into `Application Support/LiteRTModels` on the first run (resumable;
progress through `onDownloadProgress:`), then loads the engine. A request on a model that is not
loaded yet does the same before it answers. `prewarm()` on the session starts it in the background.

The complete app is [Samples/LiteRTLMDemo](Samples/LiteRTLMDemo): a streaming chat, a structured-output
screen, and a self-test that prints what the checks below measure on a device.

## Structured output

```swift
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

let plan = try await session.respond(to: "Plan a one-day trip in Kyoto.", generating: TripPlan.self)
print(plan.content.title, plan.content.stops.map(\.place))
```

The schema goes to LiteRT-LM's constrained decoding, so the reply is JSON of the schema's shape with
values inside the `@Guide` ranges and choices. Nested types and arrays work as declared. With
`includeSchemaInPrompt: true` (the default) the backend adds one line to the prompt that names the
top-level keys; the schema itself stays out of the prompt. `respond(to:schema:)` takes a
`GenerationSchema` built at run time the same way.

## Streaming

```swift
let task = Task {
  for try await snapshot in session.streamResponse(to: "Tell me about Kyoto.") {
    print(snapshot.content)  // the text so far
  }
}
// To stop, cancel the task. The backend stops the generation in LiteRT-LM.
task.cancel()
```

```swift
for try await snapshot in session.streamResponse(to: "Plan a one-day trip in Kyoto.", generating: TripPlan.self) {
  let partial = snapshot.content  // TripPlan.PartiallyGenerated
  print(partial.title ?? "…", partial.stops?.count ?? 0)
}
```

Cancelling the task that reads a stream ends the loop without an error. Cancelling a `respond` call
throws `CancellationError`. Either way the next request on the same model answers.

## Your own model file

```swift
// A .litertlm on disk: bundled, sideloaded or fine-tuned.
let local = LiteRTLMLanguageModel(modelFileURL: fileURL)

// Any .litertlm on Hugging Face, downloaded on first use.
let hub = LiteRTLMLanguageModel(
  huggingFaceRepo: "litert-community/gemma-4-E2B-it-litert-lm",
  fileName: "gemma-4-E2B-it.litertlm")
```

Both take `backend:` (`.gpu` by default), `maxTokens:` (the context size, 2048 by default) and
`cacheDirectory:` (where LiteRT-LM writes its GPU cache; the app's Caches by default). On a Mac with
macOS 27.0 the cache for Gemma 4 E2B was 0.79 GB (2026-11-07). To convert a model to `.litertlm`,
see [hf-to-litertlm](https://github.com/john-rocky/hf-to-litertlm).

## Availability and errors

```swift
switch model.availability {
case .available:
  break
case .unavailable(.notDownloaded):
  print("The model downloads on the first request.")
case .unavailable(let reason):
  print("Not available: \(reason)")
}
```

`availability` reports `.modelFileMissing(url)`, `.notDownloaded`, `.insufficientMemory` (the catalog
model needs a device with at least 7 GB of RAM; `allowUnsafeMemory: true` skips the check) and
`.loadFailed(message)` after a load threw. A failed load is tried again on the next request.

| What happened | A request throws |
|---|---|
| No model file, or LiteRT-LM cannot load it | `GenerationError.assetsUnavailable`, with LiteRT-LM's message in the context |
| The schema is one LiteRT-LM's constrained decoding does not take | `GenerationError.unsupportedGuide` |
| The input is longer than `maxTokens` | `GenerationError.exceededContextWindowSize` |
| A structured reply does not decode, for example one cut off by `maximumResponseTokens` | `GenerationError.decodingFailure` |
| The session has tools, or the prompt has an image | `LiteRTLMLanguageModelError.toolsNotSupported` / `.unsupportedContent` |

`load()` throws `LiteRTLMLanguageModelError` (`.modelFileMissing`, `.insufficientMemory`,
`.downloadFailed`, `.loadFailed`).

## Generation options

| `GenerationOptions` | LiteRT-LM | Applied |
|---|---|---|
| all `nil` | topK 40, topP 0.95, temperature 0.8, a random seed per request | yes |
| `temperature` | the sampler's temperature | yes |
| `sampling: .greedy` | topK 1, temperature 0 | yes |
| `.random(top: k, seed:)` | topK k, at most 64 | yes |
| `.random(probabilityThreshold: p, seed:)` | topP p within the 64 most likely tokens | yes |
| `maximumResponseTokens` | the output limit of the request | yes |
| `toolCallingMode`, `maximumToolRounds` | — | no (no tool calling yet) |

## Requirements

- iOS 17+ or macOS 14+, Swift 6.3+ (AnyLanguageModel 1.2.0 needs tools version 6.3). Built and tested
  with Xcode 27.0 (Swift 6.4).
- The Metal GPU engages in an Xcode-signed app on a physical device; the iOS Simulator runs on the CPU.
- iOS and macOS only. LiteRT-LM's binaries have no Mac Catalyst or visionOS slice.
- Devices and results so far: [docs/tested-devices.md](docs/tested-devices.md).

## What resolving fetches

<!-- filled from the empty-cache resolve measurement -->

## Behavior differences

- The keys of a `@Generable` reply come in no fixed order, neither declaration order nor alphabetical,
  so a streamed value fills in its properties in a different order from one run to the next. The
  decoded value is the same either way.
- Constrained decoding enforces the shape, ranges and choices, not the content: a `String` property can
  come back empty.
- topK is at most 64, the top-k capacity LiteRT-LM v0.19.0 gives its GPU sampler. Past it, LiteRT-LM
  rebuilds the sampler with code that grows with k.
- One request at a time per model file and settings; a second request waits its turn, and cancelling it
  while it waits removes it from the queue.
- Each request starts a new LiteRT-LM conversation and sends the session's transcript as its history,
  so every turn prefills the whole conversation again.
- No tool calling or image input yet. For audio, use swift-litert-lm's `LiteRTChat`.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
