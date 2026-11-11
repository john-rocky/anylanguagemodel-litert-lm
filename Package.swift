// swift-tools-version: 6.3
//
// anylanguagemodel-litert-lm: a LiteRT-LM backend for AnyLanguageModel.
//
// `LiteRTLMLanguageModel` conforms to AnyLanguageModel's `LanguageModel`, so a
// `LanguageModelSession` can run Gemma 4 and other `.litertlm` models on the device.
// `@Generable` output goes through LiteRT-LM's constrained decoding (llguidance).
//
// The runtime comes from swift-litert-lm, and this package follows it up to the next
// minor version. A patch release can move LiteRT-LM: 0.3.0 runs v0.19.0, 0.3.1 v0.19.1.

import PackageDescription

let package = Package(
  name: "anylanguagemodel-litert-lm",
  platforms: [
    .iOS(.v17),
    .macOS(.v14),
  ],
  products: [
    .library(name: "AnyLanguageModelLiteRTLM", targets: ["AnyLanguageModelLiteRTLM"])
  ],
  dependencies: [
    .package(url: "https://github.com/huggingface/AnyLanguageModel", from: "1.2.0"),
    .package(url: "https://github.com/john-rocky/swift-litert-lm", .upToNextMinor(from: "0.3.0")),
  ],
  targets: [
    .target(
      name: "AnyLanguageModelLiteRTLM",
      dependencies: [
        .product(name: "AnyLanguageModel", package: "AnyLanguageModel"),
        .product(name: "LiteRTLM", package: "swift-litert-lm"),
        .product(name: "LiteRTFoundation", package: "swift-litert-lm"),
      ]
    ),
    .testTarget(
      name: "AnyLanguageModelLiteRTLMTests",
      dependencies: [
        "AnyLanguageModelLiteRTLM",
        .product(name: "AnyLanguageModel", package: "AnyLanguageModel"),
        .product(name: "LiteRTLM", package: "swift-litert-lm"),
      ],
      // New Xcode 27 app projects turn this on; the README examples compile under it here.
      swiftSettings: [.enableUpcomingFeature("MemberImportVisibility")]
    ),
  ]
)
