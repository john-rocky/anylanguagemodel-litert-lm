import AnyLanguageModel
import AnyLanguageModelLiteRTLM
import SwiftUI

/// Generates a `@Generable` value and shows it while it fills in.
struct StructuredView: View {
  enum Kind: String, CaseIterable, Identifiable {
    case trip = "Trip plan"
    case cafe = "Cafe review"
    var id: Self { self }

    var defaultPrompt: String {
      switch self {
      case .trip: return "Plan a one-day trip in Kyoto."
      case .cafe: return "Invent a small cafe and review it."
      }
    }
  }

  @Environment(ModelHolder.self) private var holder
  @State private var kind: Kind = .trip
  @State private var prompt = Kind.trip.defaultPrompt
  @State private var trip: TripPlan.PartiallyGenerated?
  @State private var cafe: CafeReview.PartiallyGenerated?
  @State private var generation: Task<Void, Never>?
  @State private var seconds: Double?
  @State private var failure: String?

  var body: some View {
    NavigationStack {
      ScrollView {
        VStack(alignment: .leading, spacing: 16) {
          Picker("Type", selection: $kind) {
            ForEach(Kind.allCases) { Text($0.rawValue).tag($0) }
          }
          .pickerStyle(.segmented)
          .onChange(of: kind) {
            prompt = kind.defaultPrompt
            trip = nil
            cafe = nil
            seconds = nil
            failure = nil
          }

          TextField("Prompt", text: $prompt, axis: .vertical)
            .lineLimit(1...4)
            .textFieldStyle(.roundedBorder)

          HStack {
            if generation == nil {
              Button("Generate", action: generate)
                .buttonStyle(.borderedProminent)
                .disabled(prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || holder.isBusy)
            } else {
              Button("Stop") { generation?.cancel() }
                .buttonStyle(.bordered)
            }
            if let seconds {
              Text(String(format: "%.1f s", seconds)).foregroundStyle(.secondary)
            }
          }

          if let status = holder.status {
            Text(status).font(.footnote).foregroundStyle(.secondary)
          }
          if let failure {
            Text(failure).foregroundStyle(.red)
          }

          switch kind {
          case .trip:
            if let trip { TripPlanCard(plan: trip) }
          case .cafe:
            if let cafe { CafeReviewCard(review: cafe) }
          }
        }
        .padding()
      }
      .navigationTitle("Structured")
    }
  }

  private func generate() {
    trip = nil
    cafe = nil
    seconds = nil
    failure = nil
    let prompt = self.prompt
    let kind = self.kind
    // One request per session: a structured answer does not need the earlier ones.
    let session = LanguageModelSession(model: holder.model)
    generation = Task {
      do {
        try await holder.prepare()
        let start = Date()
        switch kind {
        case .trip:
          for try await snapshot in session.streamResponse(to: prompt, generating: TripPlan.self) {
            trip = snapshot.content
          }
        case .cafe:
          for try await snapshot in session.streamResponse(to: prompt, generating: CafeReview.self) {
            cafe = snapshot.content
          }
        }
        seconds = Date().timeIntervalSince(start)
      } catch is CancellationError {
      } catch {
        failure = error.localizedDescription
      }
      generation = nil
    }
  }
}

private struct TripPlanCard: View {
  let plan: TripPlan.PartiallyGenerated

  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      Text(plan.title ?? "…").font(.headline)
      if let start = plan.start {
        StopRow(label: "Start", stop: start)
      }
      // A VStack, not a List or a LazyVStack, so every stop shows as it arrives.
      VStack(alignment: .leading, spacing: 6) {
        ForEach(Array((plan.stops ?? []).enumerated()), id: \.offset) { index, stop in
          StopRow(label: "Stop \(index + 1)", stop: stop)
        }
      }
      if let needsCar = plan.needsCar {
        Text(needsCar ? "Needs a car" : "No car needed").foregroundStyle(.secondary)
      }
    }
    .padding()
    .frame(maxWidth: .infinity, alignment: .leading)
    .background(Color.secondary.opacity(0.1))
    .clipShape(RoundedRectangle(cornerRadius: 12))
  }
}

private struct StopRow: View {
  let label: String
  let stop: Stop.PartiallyGenerated

  var body: some View {
    VStack(alignment: .leading, spacing: 2) {
      Text("\(label): \(stop.place ?? "…")")
      Text("\(stop.kind ?? "…") · \(stop.minutes.map { "\($0) min" } ?? "…")")
        .font(.footnote)
        .foregroundStyle(.secondary)
    }
  }
}

private struct CafeReviewCard: View {
  let review: CafeReview.PartiallyGenerated

  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      Text(review.name ?? "…").font(.headline)
      Text("Rating: \(review.rating.map(String.init) ?? "…") / 5")
      Text("Price: \(review.price ?? "…")")
      if let hasWifi = review.hasWifi {
        Text(hasWifi ? "Free wifi" : "No wifi")
      }
    }
    .padding()
    .frame(maxWidth: .infinity, alignment: .leading)
    .background(Color.secondary.opacity(0.1))
    .clipShape(RoundedRectangle(cornerRadius: 12))
  }
}
