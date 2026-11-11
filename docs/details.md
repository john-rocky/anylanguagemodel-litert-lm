# Details

## Compared with the Foundation Models route

On iOS 27 and macOS 27, in an app built with Swift 6.4, there is another way to reach LiteRT-LM from
AnyLanguageModel: `FoundationLanguageModel` can wrap the Foundation Models backend of
[swift-litert-lm](https://github.com/john-rocky/swift-litert-lm) (`LiteRTLanguageModel`), which steers
structured output with the prompt only and does not read `GenerationOptions`. This package conforms to
AnyLanguageModel's `LanguageModel` directly: it builds for iOS 17 and macOS 14 and later, enforces the
`@Generable` schema with constrained decoding (llguidance), and applies `GenerationOptions`.

## Generation options

| `GenerationOptions` | LiteRT-LM | Applied |
|---|---|---|
| all `nil` | topK 40, topP 0.95, temperature 0.8, a random seed per request | yes |
| `temperature` | the sampler's temperature | yes |
| `sampling: .greedy` | topK 1, temperature 0 | yes |
| `.random(top: k, seed:)` | topK k, at most 64 | yes |
| `.random(probabilityThreshold: p, seed:)` | topP p within the 64 most likely tokens | yes |
| `maximumResponseTokens` | the output limit of the request | yes |
| `toolCallingMode` | `.disallowed` lets a session with tools through, without calling them; any other value throws `toolsNotSupported` | partly |
| `maximumToolRounds` | — | no (no tool calling yet) |

For repeatable output, pass `seed:` to `.random(top:seed:)` or `.random(probabilityThreshold:seed:)`;
`.greedy` gave the same bytes in two runs on a Mac (2026-11-07).

## What resolving fetches

Resolving a package that depends on this package (by path) and on AnyLanguageModel, with an empty
SwiftPM cache and no git-lfs on the PATH (2026-11-07, macOS 27.0, Xcode 27.0 (27A266a), Swift 6.4):

| What | From | Size |
|---|---|---|
| 7 git repositories: swift-syntax, swift-collections, AnyLanguageModel, swift-litert-lm, swift-http-types, JSONSchema, EventSource | GitHub | 72.0 MB (swift-syntax 41.5 MB, swift-collections 22.5 MB) |
| The LiteRT-LM v0.19.0 runtime: `CLiteRTLM.xcframework.zip` (iOS) and `CLiteRTLM_mac.xcframework.zip` (macOS) | LiteRT-LM's GitHub release | 95.8 MB + 47.0 MB |
| A swift-syntax prebuilt for macros (`swift package` only; Xcode builds swift-syntax from source) | download.swift.org | 20.1 MB |

The LiteRT-LM repository itself is not cloned, so resolving needs no git-lfs. The git sizes are the
packs of the clones. A dependency by URL also clones this repository.

## Behavior differences

- The properties of a `@Generable` reply are generated in alphabetical order, not declaration order:
  the backend sends the schema with sorted keys, and LiteRT-LM's constrained decoding follows the
  schema's order, so a streamed value fills in alphabetically. `rawContent.jsonString` lists the keys in
  no fixed order, because AnyLanguageModel's `GeneratedContent(json:)` keeps an object's keys in a
  dictionary. The decoded value is the same either way.
- Constrained decoding enforces the shape, ranges and choices, not the content: a `String` property can
  come back empty.
- topK is at most 64, the default top-k capacity of LiteRT-LM v0.19.0's GPU sampler. Past it, LiteRT-LM
  recreates the sampler for the larger k, with GPU code that grows with k.
- One request at a time per model file and settings; a second request waits its turn, and cancelling it
  while it waits removes it from the queue.
- Each request starts a new LiteRT-LM conversation and sends the session's transcript as its history,
  so the whole transcript goes to the model on every turn; nothing is kept between requests.
- No tool calling or image input yet. For audio, use swift-litert-lm's `LiteRTChat`.

## Model arguments

`backend:` is `.gpu` by default. `Backend` comes with this module, so it needs no extra import. On a Mac with macOS 27.0, the GPU cache for Gemma 4 E2B was 0.79 GB on 2026-11-07.
