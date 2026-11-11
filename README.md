# AnyLanguageModel LiteRT-LM

A Swift package that adds Gemma 4 and other `.litertlm` models, running on-device on iPhone and Mac, to
[AnyLanguageModel](https://github.com/huggingface/AnyLanguageModel), Hugging Face's drop-in replacement for Apple's Foundation Models API.
You write the same `LanguageModelSession` and `@Generable` code, and it builds for iOS 17 and macOS 14 or later, older versions than Apple's own framework supports. I've run it on iOS 27.2 and macOS 27.0.
Google's [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) runtime does the work, and its constrained decoding enforces your `@Generable` schema.

On 2026-11-07, Gemma 4 E2B streamed its reply at 88.6 tokens per second on an iPhone 18 Pro with LiteRT-LM v0.19.0.
A one-day trip plan built from nested `@Generable` types came back in 3.6 to 3.9 seconds.
Tools and images are not supported yet.

Community package, not affiliated with Google, Apple or Hugging Face.

## Quickstart

You need Swift 6.3 or newer. On iOS, the GPU needs a physical device and an Xcode-signed app. The iOS Simulator runs on the CPU.

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

`load()` downloads the model into `Application Support/LiteRTModels` on the first run, then loads the engine.
Interrupted downloads resume. To show progress, pass `onDownloadProgress:` when you create the model. You can also skip `load()`: the
first request downloads and loads the model before it answers. To load it ahead of time, in the background,
call `prewarm()` on the session.

`.gemma4_E2B` is `gemma-4-E2B-it.litertlm` from
[litert-community/gemma-4-E2B-it-litert-lm](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm).
It needs a device with at least 7 GB of RAM.

A complete sample app is in [Samples/LiteRTLMDemo](Samples/LiteRTLMDemo): a streaming chat, a structured-output
screen, and a self-test that runs the checks listed in [docs/tested-devices.md](docs/tested-devices.md)
on a device and prints the results. Open `Samples/LiteRTLMDemo/LiteRTLMDemo.xcodeproj`, set your team
under Signing, and run it on an iPhone. The Xcode project is generated from `project.yml` with XcodeGen.

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

let plan = try await session.respond(
  to: "Plan a one-day trip in Kyoto.",
  generating: TripPlan.self)
print(plan.content.title, plan.content.stops.map(\.place))
```

LiteRT-LM's constrained decoding enforces the schema, so the values stay inside the `@Guide` ranges
and choices. Nested types and arrays work as declared. By default,
`includeSchemaInPrompt: true` makes the backend add one line to the prompt that names the
top-level keys. The schema itself stays out of the prompt. To build a schema at run time,
pass a `GenerationSchema` to `respond(to:schema:)`.

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
for try await snapshot in session.streamResponse(
  to: "Plan a one-day trip in Kyoto.",
  generating: TripPlan.self
) {
  let partial = snapshot.content  // TripPlan.PartiallyGenerated
  print(partial.title ?? "…", partial.stops?.count ?? 0)
}
```

Cancelling the task that reads a stream ends the loop without an error. Cancelling a `respond` call
throws `CancellationError`. Either way, the model stays usable for the next request.

## Your own model file

```swift
// A .litertlm on disk: bundled, sideloaded or fine-tuned.
let local = LiteRTLMLanguageModel(modelFileURL: fileURL)

// Any public .litertlm on Hugging Face, downloaded on first use.
let hub = LiteRTLMLanguageModel(
  huggingFaceRepo: "litert-community/gemma-4-E2B-it-litert-lm",
  fileName: "gemma-4-E2B-it.litertlm")
```

The download sends no Hugging Face token, so files in private and gated repos fail to download. Download those yourself and use `modelFileURL:`.

Both initializers also take these arguments:

| Argument | What it sets | Default |
|---|---|---|
| `backend:` | The compute backend | `.gpu` |
| `maxTokens:` | The context size | 2048 |
| `cacheDirectory:` | Where LiteRT-LM writes its GPU cache | The app's Caches |

To convert a model to `.litertlm`, see [hf-to-litertlm](https://github.com/john-rocky/hf-to-litertlm).

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

`availability` reports `.modelFileMissing(url)`, `.notDownloaded`, `.insufficientMemory` or, after a
load threw, `.loadFailed(message)`. `allowUnsafeMemory: true` skips the memory check. A failed load is tried again on the next request.

Most errors are AnyLanguageModel's `GenerationError` or this package's `LiteRTLMLanguageModelError`. A cancelled `respond` throws `CancellationError`, and other runtime failures arrive as swift-litert-lm's `LiteRTLMError`.

| What happened | A request throws |
|---|---|
| No model file, or LiteRT-LM cannot load it | `GenerationError.assetsUnavailable`, with the error message in the context |
| LiteRT-LM's constrained decoding can't handle the schema | `GenerationError.unsupportedGuide` |
| The input is longer than `maxTokens` | `GenerationError.exceededContextWindowSize` |
| A structured reply does not decode, for example one cut off by `maximumResponseTokens` | `GenerationError.decodingFailure` |
| The session has tools, or the prompt has an image | `LiteRTLMLanguageModelError.toolsNotSupported` / `.unsupportedContent` |

`load()` throws `LiteRTLMLanguageModelError`: `.modelFileMissing`, `.insufficientMemory`,
`.downloadFailed` or `.loadFailed`.

## Requirements

- iOS 17+ or macOS 14+. Swift 6.3+, which AnyLanguageModel 1.2.0 needs. Built and tested
  with Xcode 27.0 and Swift 6.4.
- iOS and macOS only. LiteRT-LM's binaries have no Mac Catalyst or visionOS slice.
- Devices and results so far: [docs/tested-devices.md](docs/tested-devices.md).

## Documentation

- [Details](docs/details.md): how `GenerationOptions` map to LiteRT-LM, what resolving the package downloads, behavior differences, and how this package differs from the Foundation Models backend in [swift-litert-lm](https://github.com/john-rocky/swift-litert-lm)
- [Tested devices](docs/tested-devices.md)

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
