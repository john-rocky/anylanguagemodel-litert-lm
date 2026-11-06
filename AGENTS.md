# AGENTS.md — for coding agents putting a LiteRT-LM model behind AnyLanguageModel

Read this first if someone asked you to run Gemma or another `.litertlm` model through
[AnyLanguageModel](https://github.com/huggingface/AnyLanguageModel)'s `LanguageModelSession` on an
iPhone or a Mac. This file routes you to the right API and the file that shows it; the README has
the code.

**What this is.** A LiteRT-LM backend for AnyLanguageModel 1.2: `LiteRTLMLanguageModel` conforms to
`LanguageModel`, so `LanguageModelSession`, `@Generable`, `streamResponse` and `GenerationOptions`
work unchanged. `@Generable` output goes through LiteRT-LM's constrained decoding. The runtime
comes from [swift-litert-lm](https://github.com/john-rocky/swift-litert-lm) 0.3.x, which runs
LiteRT-LM v0.19.0. Community package, not affiliated with Google, Apple or Hugging Face.

## Route

| Task | API | Where |
|---|---|---|
| Text from Gemma 4 E2B in a `LanguageModelSession` | `LiteRTLMLanguageModel(.gemma4_E2B)`, then `LanguageModelSession(model:)` and `respond(to:)` | README "Quickstart" |
| A typed value from a `@Generable` type, nested types and arrays included | `session.respond(to:generating:)` | README "Structured output" |
| A schema built at run time | `session.respond(to:schema:)` with a `GenerationSchema` | README "Structured output" |
| Streamed text or a partial value, with a Stop button | `streamResponse(to:)` / `streamResponse(to:generating:)`; Stop = cancel the reading task | README "Streaming", [Samples/LiteRTLMDemo/Sources/ChatView.swift](Samples/LiteRTLMDemo/Sources/ChatView.swift) |
| A `.litertlm` on disk (bundled, sideloaded, fine-tuned) | `LiteRTLMLanguageModel(modelFileURL:)` | README "Your own model file" |
| Any `.litertlm` on Hugging Face | `LiteRTLMLanguageModel(huggingFaceRepo:fileName:)` | README "Your own model file" |
| Download progress, "not downloaded", load errors on screen | `availability`, `load()`, `onDownloadProgress:` | README "Availability and errors", [Samples/LiteRTLMDemo/Sources/LiteRTLMDemoApp.swift](Samples/LiteRTLMDemo/Sources/LiteRTLMDemoApp.swift) |
| A complete app to copy (chat, structured output, device self-test) | XcodeGen project, iOS and macOS | [Samples/LiteRTLMDemo](Samples/LiteRTLMDemo) |
| Chat with images or audio, or no AnyLanguageModel at all | swift-litert-lm's `LiteRTChat` | [swift-litert-lm](https://github.com/john-rocky/swift-litert-lm) |
| Convert a model to `.litertlm` | — | [hf-to-litertlm](https://github.com/john-rocky/hf-to-litertlm) |

## Rules that fail on a real device when broken

1. **Pin a tag.** `.package(url: "https://github.com/john-rocky/anylanguagemodel-litert-lm", from: "0.1.0")`,
   next to `.package(url: "https://github.com/huggingface/AnyLanguageModel", from: "1.2.0")`. This package
   follows swift-litert-lm up to its next minor version (0.3.x = LiteRT-LM v0.19.0).
2. **The Metal GPU engages only in an Xcode-signed app on a physical device.** The iOS Simulator
   runs on the CPU. Do not report simulator speed as device speed.
3. **Do not commit or bundle the model.** `LiteRTLMLanguageModel(.gemma4_E2B)` downloads about 2.6 GB
   into `Application Support/LiteRTModels` on first use.
4. **topK is at most 64; do not pass a larger k some other way.** LiteRT-LM v0.19.0 builds its GPU
   sampler's code in proportion to k. With k = Int32.max a Mac test process passed 10 GB within
   seconds (2026-11-07, macOS 27.0); on an iPhone that is the app being killed.
5. **After an error or a cancel, keep the model and the session.** Each request runs in a new
   LiteRT-LM conversation, so a failed or cancelled request does not break the next one. On a Mac
   (macOS 27.0, LiteRT-LM v0.19.0) the next request answered after each of: a missing file, three
   broken files, an unsupported schema, a cancelled `respond` and a cancelled stream.
6. **One generation at a time per model.** A second request waits its turn. Each distinct
   combination of file, backend, `maxTokens` and `cacheDirectory` loads its own engine and its
   own memory, so do not make variants of one model to run requests in parallel.
7. **Do not rely on key order in a `@Generable` reply.** It is neither declaration order nor
   alphabetical, and it changes between runs; a streamed value fills in a different order each time.
8. **No tool calling or image input yet.** A session with tools throws
   `LiteRTLMLanguageModelError.toolsNotSupported` (or set `GenerationOptions.toolCallingMode` to
   `.disallowed`); an image throws `.unsupportedContent`. Audio has no path in AnyLanguageModel; use
   `LiteRTChat`.
9. **A macOS app needs `@executable_path/../Frameworks` in `LD_RUNPATH_SEARCH_PATHS`.** LiteRT-LM's
   macOS runtime is a dylib that Xcode embeds in `Contents/Frameworks`. A multiplatform target that
   keeps the iOS value (`@executable_path/Frameworks`) builds, then fails at launch with
   `Library not loaded: @rpath/libCLiteRTLM_mac.dylib`. The sample's `project.yml` sets it.
10. **No `import LiteRTFoundation` is needed.** `.gemma4_E2B`, `.gpu` and `.cpu()` resolve through this
    package. The type name `LiteRTModel` itself is not visible without importing `LiteRTFoundation`.
11. **Numbers stay with their device and runtime.** Results so far are in
    [docs/tested-devices.md](docs/tested-devices.md). Do not extrapolate to other devices or runtime
    versions.
12. iOS 17+ or macOS 14+; Swift tools 6.3 or later (built with Xcode 27.0, Swift 6.4). iOS and macOS
    only: LiteRT-LM's binaries have no Mac Catalyst or visionOS slice.

## Not this package

- Android → [hfmodels-android](https://github.com/john-rocky/hfmodels-android).
- Apple's Foundation Models API with an open model (iOS 27) → swift-litert-lm's `LiteRTLanguageModel`.

Maintainer: john-rocky (GitHub). Issues: https://github.com/john-rocky/anylanguagemodel-litert-lm/issues
