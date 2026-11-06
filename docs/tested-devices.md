# Tested devices

Each row is one run of this package's checks with Gemma 4 E2B (`gemma-4-E2B-it.litertlm`,
2,588,147,712 bytes) on the GPU backend:

- **Integration tests**: `swift test` with `LITERT_TEST_MODEL` set (`Tests/AnyLanguageModelLiteRTLMTests/IntegrationTests.swift`).
- **Sample self-test**: the sample app launched with `LITERT_SELFTEST=all` (`Samples/LiteRTLMDemo/Sources/SelfTest.swift`).

| Device | OS | LiteRT-LM (swift-litert-lm) | Check | respond | stream | flat | nested | failure paths | Log | Date |
|---|---|---|---|---|---|---|---|---|---|---|
| Mac (Mac16,9) | macOS 27.0 (26A428) | v0.19.0 (0.3.0) | integration tests | PASS | PASS | 3/3 | 3/3 | 6/6 | `integration_4.log` | 2026-11-07 |
| Mac (Mac16,9) | macOS 27.0 (26A428) | v0.19.0 (0.3.0) | sample self-test | PASS | PASS | 3/3 | 3/3 | 6/6 | `mac_selftest.log` | 2026-11-07 |
| iPhone 18 Pro (iPhone19,2) | iOS 27.2 | v0.19.0 (0.3.0) | sample self-test | not run yet | | | | | | |

What passes:

- **respond / stream**: the reply ends at the end-of-sequence token with text that is not empty
  (at least one token, and prompt plus reply under the 2048-token context).
- **flat**: `respond(to: "Invent a small cafe and review it.", generating: CafeReview.self)` returns a
  value with `rating` in 1...5 and `price` one of cheap, moderate, expensive.
- **nested**: `respond(to: "Plan a one-day trip in Kyoto.", generating: TripPlan.self)` returns a value
  with 1 to 4 `stops`, and every stop's `minutes` in 15...240 and `kind` one of the five choices.
- **failure paths**: (a) a missing file, (b1) 1 MiB of random bytes, (b2) the first 64 MiB of the real
  model, (b3) a directory, (c) a `GenerationSchema` with a `$ref` to nothing, (d) cancelling the task
  that reads a stream. Each one throws an error with a description (for (d): the task ends within
  60 s), the process does not crash, and a new session on the real model answers afterwards.

The types are `CafeReview`, `TripPlan` and `Stop` from `Samples/LiteRTLMDemo/Sources/Types.swift`.
No speed is recorded here for the Mac.
