// LiteRTLMDemo: Gemma 4 E2B through AnyLanguageModel's LanguageModelSession.
//
// Two screens: Chat (streamed text) and Structured (a @Generable type streamed as it
// fills in). Launched with LITERT_SELFTEST set, the app runs a headless self-test instead
// and prints its results (see SelfTest.swift).

import AnyLanguageModel
import AnyLanguageModelLiteRTLM
import Foundation
import Observation
import SwiftUI

@main
struct LiteRTLMDemoApp: App {
  @State private var holder = ModelHolder()

  init() {
    // Unbuffered stdout, so a device console sees self-test lines as they are printed.
    setvbuf(stdout, nil, _IONBF, 0)
  }

  var body: some Scene {
    WindowGroup {
      if SelfTest.isRequested {
        SelfTestView()
      } else {
        TabView {
          ChatView()
            .tabItem { Label("Chat", systemImage: "bubble.left.and.bubble.right") }
          StructuredView()
            .tabItem { Label("Structured", systemImage: "list.bullet.rectangle") }
        }
        .environment(holder)
      }
    }
  }
}

/// The model both screens use, with its download and load state.
@MainActor @Observable
final class ModelHolder {
  enum Phase: Equatable {
    case idle
    case downloading(Double)
    case loading
    case ready
    case failed(String)
  }

  let model: LiteRTLMLanguageModel
  private(set) var phase: Phase = .idle

  init() {
    let (updates, continuation) = AsyncStream.makeStream(
      of: Double.self, bufferingPolicy: .bufferingNewest(1))
    // Catalog model: downloads on first use into Application Support/LiteRTModels.
    model = LiteRTLMLanguageModel(.gemma4_E2B) { progress in
      continuation.yield(progress.fraction)
    }
    Task { [weak self] in
      for await fraction in updates {
        guard let self, case .downloading = self.phase else { continue }
        self.phase = fraction < 1 ? .downloading(fraction) : .loading
      }
    }
  }

  /// Downloads the model file if needed and loads the engine. Requests would do this
  /// themselves; calling it first lets the screen show what is happening.
  func prepare() async throws {
    if phase == .ready { return }
    if case .unavailable(.notDownloaded) = model.availability {
      phase = .downloading(0)
    } else {
      phase = .loading
    }
    do {
      try await model.load()
      phase = .ready
    } catch {
      phase = .failed(error.localizedDescription)
      throw error
    }
  }

  /// One line about the model's state, or nil when there is nothing to say.
  var status: String? {
    switch phase {
    case .downloading(let fraction):
      return "Downloading Gemma 4 E2B… \(Int(fraction * 100))%"
    case .loading:
      return "Loading the model…"
    case .failed(let message):
      return message
    case .ready:
      return nil
    case .idle:
      if case .unavailable(let reason) = model.availability { return reason.message }
      return nil
    }
  }

  var isBusy: Bool {
    switch phase {
    case .downloading, .loading: return true
    default: return false
    }
  }
}

extension LiteRTLMLanguageModel.UnavailableReason {
  var message: String {
    switch self {
    case .notDownloaded:
      return "Gemma 4 E2B (about 2.6 GB) downloads when you send the first message."
    case .modelFileMissing(let url):
      return "No model file at \(url.path)."
    case .insufficientMemory(let available, let required):
      let format = { (bytes: Int64) in ByteCountFormatter.string(fromByteCount: bytes, countStyle: .memory) }
      return "This device has \(format(available)) of RAM; Gemma 4 E2B needs \(format(required))."
    case .loadFailed(let details):
      return "The model did not load: \(details)"
    }
  }
}
