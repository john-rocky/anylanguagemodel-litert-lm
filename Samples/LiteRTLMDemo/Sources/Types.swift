import AnyLanguageModel

// The types the Structured screen and the self-test generate.

@Generable(description: "A short review of a cafe")
struct CafeReview {
  @Guide(description: "The cafe's name")
  var name: String
  @Guide(description: "Rating from 1 to 5", .range(1...5))
  var rating: Int
  @Guide(description: "Price level", .anyOf(["cheap", "moderate", "expensive"]))
  var price: String
  @Guide(description: "Whether the cafe has free wifi")
  var hasWifi: Bool
}

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
