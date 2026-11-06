import AnyLanguageModel
import Foundation

/// The JSON schema and the prompt line for structured output, and the decoding of the reply.
enum StructuredOutput {
  /// The schema as LiteRT-LM's constrained decoding takes it.
  ///
  /// The keys are sorted: `JSONEncoder` writes a schema's keys in a different order on each
  /// call, and the model generates the properties in the order the schema lists them, so
  /// sorted keys give the same schema, and the same property order, every time.
  static func schemaJSON(for schema: GenerationSchema) throws -> String {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    return String(decoding: try encoder.encode(schema), as: UTF8.self)
  }

  /// The line the backend appends to the prompt when `includeSchemaInPrompt` is true:
  /// the top-level keys, in the order the schema lists them.
  ///
  /// The schema itself is not put in the prompt. Under constrained decoding a model tends
  /// to copy schema text into the reply, and the grammar then turns the copy into JSON of
  /// the right shape with the wrong content.
  static func keyLine(forSchemaJSON json: String) -> String {
    let keys = topLevelKeys(ofSchemaJSON: json)
    switch keys.count {
    case 0:
      return "Reply in JSON."
    case 1:
      return "Reply in JSON with the key \(keys[0])."
    default:
      let head = keys.dropLast().joined(separator: ", ")
      return "Reply in JSON with the keys \(head) and \(keys[keys.count - 1])."
    }
  }

  /// The keys of the root object, following a root `$ref` into `$defs`; sorted.
  static func topLevelKeys(ofSchemaJSON json: String) -> [String] {
    guard let schema = (try? JSONSerialization.jsonObject(with: Data(json.utf8))) as? [String: Any]
    else { return [] }
    var root = schema
    if let ref = schema["$ref"] as? String, ref.hasPrefix("#/$defs/"),
      let defs = schema["$defs"] as? [String: Any],
      let target = defs[String(ref.dropFirst("#/$defs/".count))] as? [String: Any]
    {
      root = target
    }
    guard let properties = root["properties"] as? [String: Any] else { return [] }
    return properties.keys.sorted()
  }

  /// The prompt with the key line, when the request asks for it.
  static func message(prompt: String, schemaJSON: String?, includeSchemaInPrompt: Bool) -> String {
    guard let schemaJSON, includeSchemaInPrompt else { return prompt }
    return prompt + "\n\n" + keyLine(forSchemaJSON: schemaJSON)
  }

  /// Decodes a complete reply. A reply that is not complete JSON (for example one cut off by
  /// `maximumResponseTokens`) is a decoding failure, not a value with missing parts.
  static func decode<Content: Generable>(
    _ text: String, as type: Content.Type
  ) throws -> (content: Content, raw: GeneratedContent) {
    guard (try? JSONSerialization.jsonObject(with: Data(text.utf8), options: [.fragmentsAllowed])) != nil
    else {
      throw ErrorMapping.decodingError(
        "The model's reply is not complete JSON (\(text.count) characters): \(text.prefix(200))")
    }
    let raw: GeneratedContent
    do {
      raw = try GeneratedContent(json: text)
    } catch {
      throw ErrorMapping.decodingError("The model's reply did not parse as JSON: \(error)")
    }
    do {
      return (try Content(raw), raw)
    } catch {
      throw ErrorMapping.decodingError(
        "The model's reply did not decode as \(Content.self): \(error). Reply: \(text.prefix(200))")
    }
  }

  /// The raw content of a reply so far, or `nil` while it does not parse yet.
  static func partialRaw(_ text: String, structured: Bool) -> GeneratedContent? {
    guard structured else { return GeneratedContent(text) }
    guard let raw = try? GeneratedContent(json: text) else { return nil }
    // `GeneratedContent(json:)` falls back to the text itself as a string when the JSON
    // cannot be completed yet; that is not a partial value.
    if case .string(let value) = raw.kind,
      value == text.trimmingCharacters(in: .whitespacesAndNewlines)
    {
      return nil
    }
    return raw
  }
}

/// Decodes the partial form of `Value`, or returns `nil`.
func partiallyGenerated<Value: Generable>(
  _: Value.Type, from raw: GeneratedContent
) -> Value.PartiallyGenerated? {
  try? Value.PartiallyGenerated(raw)
}
