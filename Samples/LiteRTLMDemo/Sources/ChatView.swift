import AnyLanguageModel
import AnyLanguageModelLiteRTLM
import SwiftUI

/// A chat on one `LanguageModelSession`: the session keeps the transcript, and each reply
/// streams in as it is generated.
struct ChatView: View {
  @Environment(ModelHolder.self) private var holder
  @State private var session: LanguageModelSession?
  @State private var turns: [Turn] = []
  @State private var input = ""
  @State private var reply: Task<Void, Never>?

  struct Turn: Identifiable {
    enum Role { case user, model }
    let id = UUID()
    let role: Role
    var text: String
    var failed = false
  }

  var body: some View {
    NavigationStack {
      VStack(spacing: 0) {
        if let status = holder.status {
          Text(status)
            .font(.footnote)
            .foregroundStyle(.secondary)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal)
            .padding(.vertical, 6)
        }
        ScrollViewReader { proxy in
          ScrollView {
            // A VStack, not a LazyVStack: a lazy stack can leave the list blank after a
            // very long message.
            VStack(alignment: .leading, spacing: 12) {
              ForEach(turns) { turn in
                bubble(turn).id(turn.id)
              }
            }
            .padding()
          }
          .onChange(of: turns.last?.text) {
            if let last = turns.last { proxy.scrollTo(last.id, anchor: .bottom) }
          }
        }
        Divider()
        inputBar
      }
      .navigationTitle("Chat")
      .toolbar {
        ToolbarItem(placement: .primaryAction) {
          Button("New chat", action: newChat).disabled(reply != nil)
        }
      }
    }
  }

  private var inputBar: some View {
    HStack(alignment: .bottom) {
      TextField("Message", text: $input, axis: .vertical)
        .lineLimit(1...5)
        .textFieldStyle(.roundedBorder)
        .onSubmit(send)
      if reply == nil {
        Button("Send", action: send)
          .disabled(input.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || holder.isBusy)
      } else {
        Button("Stop") { reply?.cancel() }
      }
    }
    .padding()
  }

  private func bubble(_ turn: Turn) -> some View {
    let isUser = turn.role == .user
    return HStack {
      if isUser { Spacer(minLength: 40) }
      Text(turn.text.isEmpty ? "…" : turn.text)
        .textSelection(.enabled)
        .padding(10)
        .foregroundStyle(turn.failed ? Color.red : Color.primary)
        .background(isUser ? Color.accentColor.opacity(0.2) : Color.secondary.opacity(0.15))
        .clipShape(RoundedRectangle(cornerRadius: 12))
      if !isUser { Spacer(minLength: 40) }
    }
  }

  private func send() {
    let prompt = input.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !prompt.isEmpty, reply == nil else { return }
    input = ""
    turns.append(Turn(role: .user, text: prompt))
    turns.append(Turn(role: .model, text: ""))
    let index = turns.count - 1
    let session = self.session ?? LanguageModelSession(model: holder.model)
    self.session = session
    reply = Task {
      do {
        try await holder.prepare()
        for try await snapshot in session.streamResponse(to: prompt) {
          turns[index].text = snapshot.content
        }
      } catch is CancellationError {
        // Stop keeps the text so far.
      } catch {
        turns[index].text = error.localizedDescription
        turns[index].failed = true
      }
      reply = nil
    }
  }

  /// A new session on the same model: the engine stays loaded.
  private func newChat() {
    session = nil
    turns = []
  }
}
