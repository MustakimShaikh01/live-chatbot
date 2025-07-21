import random
import json
import nltk
import string

from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load intents
with open("intents.json") as file:
    data = json.load(file)

lemmatizer = WordNetLemmatizer()

# Prepare data
all_patterns = []
all_tags = []
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        all_patterns.append(pattern)
        all_tags.append(intent["tag"])

# Preprocess text
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return lemmatizer.lemmatize(text)

cleaned_patterns = [clean_text(p) for p in all_patterns]

# Vectorize and train model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(cleaned_patterns)
y = all_tags

model = MultinomialNB()
model.fit(X, y)

# Chat function
def chatbot_response(user_input):
    cleaned = clean_text(user_input)
    X_test = vectorizer.transform([cleaned])
    predicted_tag = model.predict(X_test)[0]

    for intent in data["intents"]:
        if intent["tag"] == predicted_tag:
            return random.choice(intent["responses"])
    return "I'm not sure I understand."

# Run chatbot
def chat():
    print("Chatbot is ready! Type 'quit' to exit.
")
    while True:
        msg = input("You: ")
        if msg.lower() == "quit":
            print("Chatbot: Goodbye!")
            break
        print("Chatbot:", chatbot_response(msg))

if __name__ == "__main__":
    nltk.download("punkt")
    nltk.download("wordnet")
    chat()