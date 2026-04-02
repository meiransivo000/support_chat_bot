import random
import json
from nltk.tokenize import wordpunct_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


# Loading JSON files with training and responses
with open("training_data.json", "r", encoding="utf-8") as f:
    intents_data = json.load(f)

with open("responses.json", "r", encoding="utf-8") as f:
    responses_data = json.load(f)

responses = responses_data["responses"]

# Training lists
training_sentences = []
training_labels = []

for intent in intents_data["training_data"]:
    tag = intent["tag"]
    for pattern in intent["patterns"]:
        training_sentences.append(pattern)
        training_labels.append(tag)

# Primitive training
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(training_sentences)

model = LogisticRegression(max_iter=200)
model.fit(X, training_labels)

# Names in memory
memory = {
    "user_name": None,
    "last_intent": None,
    "last_message": None
}

# Key word detection function
def detect_keywords(text):
    tokens = wordpunct_tokenize(text.lower())
    text_lower = text.lower().strip()

    if any(word in tokens for word in ["order", "tracking", "status"]) and \
       any(word in tokens for word in ["missing", "invisible", "visible", "see", "find", "where"]):
        return "order_not_visible"

    if any(word in tokens for word in ["stolen", "lost", "missing"]):
        return "order_stolen"

    if any(word in tokens for word in ["damaged", "spilled", "ruined", "broken", "cold"]):
        return "food_damaged"

    if text_lower in ["hi", "hello", "hey", "good morning"]:
        return "greeting"

    if text_lower in ["bye", "goodbye", "exit", "quit"]:
        return "goodbye"

    return None

# Thing responsible for saving username in case they have introduced themselves
def save_name(text):
    text_lower = text.lower()
    patterns = ["my name is ", "i am ", "i'm "]

    for pattern in patterns:
        if pattern in text_lower:
            start = text_lower.find(pattern) + len(pattern)
            parts = text[start:].strip().split()
            if parts:
                memory["user_name"] = parts[0].capitalize()
                return True
    return False

# Prediction using keywords and then model do the rest
def predict_intent(user_input):
    keyword_intent = detect_keywords(user_input)
    if keyword_intent:
        return keyword_intent

    user_vector = vectorizer.transform([user_input])
    predicted = model.predict(user_vector)[0]

    probabilities = model.predict_proba(user_vector)[0]
    max_prob = max(probabilities)

    if max_prob < 0.40:
        return "unknown"

    return predicted

# In case if advice did not help, it is for real situation simulation, to like satisfy user somehow
def get_followup_response():
    match memory["last_intent"]:
        case "order_not_visible":
            return "I’m sorry that didn’t help. Please refresh the app and check your notifications. If the order is still missing, contact live support."

        case "order_stolen":
            return "I’m really sorry about that. Please report the order as missing in the app, and support can help with a refund or replacement."

        case "food_damaged":
            return "I’m sorry your order arrived in bad condition. Please send a photo through the app so support can review it and offer compensation."

        case _:
            return "Can you remind me what happened with your order?"

# exit thing
def get_response(user_input):
    text_lower = user_input.lower().strip()

    if text_lower in ["bye", "goodbye", "exit", "quit"]:
        memory["last_intent"] = "goodbye"
        return random.choice(responses["goodbye"])

    if save_name(user_input):
        return f"Nice to meet you, {memory['user_name']}. How can I help with your order today?"

    match text_lower:
        case "it still doesn't work" | "still not working" | "didn't help" | "not fixed":
            return get_followup_response()

    intent = predict_intent(user_input)

    memory["last_intent"] = intent
    memory["last_message"] = user_input

    chosen_response = random.choice(responses.get(intent, responses["unknown"]))

    if memory["user_name"] and intent not in ["greeting", "goodbye"]:
        return f"{memory['user_name']}, {chosen_response}"

    return chosen_response


def run_chatbot():
    print("FoodDeliveryBot: Hello! I’m here to help with your order. Type 'bye' to exit.")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in {"bye", "goodbye", "exit", "quit"}:
            print("Bot:", random.choice(responses["goodbye"]))
            break

        reply = get_response(user_input)
        print("Bot:", reply)


if __name__ == "__main__":
    run_chatbot()