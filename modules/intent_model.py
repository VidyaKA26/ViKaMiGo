import spacy
from spacy.training.example import Example
from spacy.util import minibatch
import random
from transformers import pipeline

class NonZeroShotIntentModel:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_trf")  # BERT-based spaCy model
        self.text_classifier = self.nlp.create_pipe("textcat", config={"architecture": "bow"})
        self.nlp.add_pipe(self.text_classifier, last=True)
        self.labels = ["Product Inquiry", "Order Inquiry", "Technical Support", "Refund Inquiry", "Catalog Inquiry", "Return Request"]
        for label in self.labels:
            self.text_classifier.add_label(label)
        
    def train_model(self, train_data, epochs=10):
        optimizer = self.nlp.begin_training()
        for epoch in range(epochs):
            random.shuffle(train_data)
            losses = {}
            for batch in minibatch(train_data, size=8):
                for text, annotations in batch:
                    example = Example.from_dict(self.nlp.make_doc(text), annotations)
                    self.nlp.update([example], drop=0.5, losses=losses)
            print(f"Epoch {epoch+1}/{epochs} - Losses: {losses}")
        
    def save_model(self, output_dir="vikamigo_data/model"):
        self.nlp.to_disk(output_dir)
        
        

class ZeroShotIntentModel:
    def __init__(self):
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    def classify_intent(self, text, candidate_labels):
        return self.classifier(text, candidate_labels)

# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report
# import pickle

# class IntentModel:
#     def __init__(self):
#         self.model = LogisticRegression()

#     def train(self, X_train, y_train):
#         self.model.fit(X_train, y_train)
#         print("Model trained successfully.")

#     def evaluate(self, X_test, y_test):
#         y_pred = self.model.predict(X_test)
#         print("Evaluation Report:\n", classification_report(y_test, y_pred))

#     def save(self, model_path, vectorizer_path, vectorizer):
#         with open(model_path, "wb") as f:
#             pickle.dump(self.model, f)
#         with open(vectorizer_path, "wb") as f:
#             pickle.dump(vectorizer, f)
#         print("Model and vectorizer saved.")

#     @staticmethod
#     def load(model_path, vectorizer_path):
#         with open(model_path, "rb") as f:
#             model = pickle.load(f)
#         with open(vectorizer_path, "rb") as f:
#             vectorizer = pickle.load(f)
#         print("Model and vectorizer loaded.")
#         return model, vectorizer
