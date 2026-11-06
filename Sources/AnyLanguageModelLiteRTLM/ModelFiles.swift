import Foundation
import LiteRTFoundation

extension LiteRTLMLanguageModel {
  /// How far the model file's download has come.
  public struct DownloadProgress: Equatable, Sendable {
    public let completedBytes: Int64
    public let totalBytes: Int64

    /// From 0 to 1; 0 while the total is unknown.
    public var fraction: Double {
      totalBytes > 0 ? min(Double(completedBytes) / Double(totalBytes), 1) : 0
    }

    public init(completedBytes: Int64, totalBytes: Int64) {
      self.completedBytes = completedBytes
      self.totalBytes = totalBytes
    }
  }
}

/// Where model files live, and their downloads. The same places and downloader as
/// swift-litert-lm's `LiteRTChat`, so an app that uses both downloads a model once.
enum ModelFiles {
  typealias ProgressHandler = @Sendable (LiteRTLMLanguageModel.DownloadProgress) -> Void

  /// Application Support/LiteRTModels, where `LiteRTChat` downloads. Not created here.
  static func defaultStorageDirectory() -> URL? {
    FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first?
      .appendingPathComponent("LiteRTModels", isDirectory: true)
  }

  /// The Caches directory, where `LiteRTChat` lets LiteRT-LM write its GPU cache.
  static func defaultCacheDirectory() -> URL? {
    FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
  }

  static func fileExists(_ url: URL) -> Bool {
    FileManager.default.fileExists(atPath: url.path)
  }

  /// Downloads a catalog model unless it is already there.
  static func downloadCatalogModel(
    _ model: LiteRTModel, storageDirectory: URL?, onProgress: ProgressHandler?
  ) async throws {
    do {
      try await LiteRTChat.ensureModel(
        model, storageDirectory: storageDirectory, onProgress: progress(onProgress))
    } catch {
      throw LiteRTLMLanguageModelError.downloadFailed(ErrorMapping.describe(error))
    }
  }

  /// Downloads `fileName` from a Hugging Face repo to `destination` unless it is already there.
  static func downloadFromHuggingFace(
    repo: String, fileName: String, revision: String, destination: URL,
    onProgress: ProgressHandler?
  ) async throws {
    guard
      let url = URL(
        string: "https://huggingface.co/\(repo)/resolve/\(revision)/\(fileName)?download=true")
    else {
      throw LiteRTLMLanguageModelError.downloadFailed(
        "No download URL for \(fileName) in \(repo) at \(revision).")
    }
    do {
      try await ModelDownloader.shared.download(
        from: url, to: destination, expectedBytes: nil, onProgress: progress(onProgress))
    } catch {
      throw LiteRTLMLanguageModelError.downloadFailed(ErrorMapping.describe(error))
    }
  }

  private static func progress(
    _ handler: ProgressHandler?
  ) -> (@Sendable (ModelDownloader.Progress) -> Void)? {
    guard let handler else { return nil }
    return { progress in
      handler(.init(completedBytes: progress.completedBytes, totalBytes: progress.totalBytes))
    }
  }
}
