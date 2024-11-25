from flask import Flask, request, jsonify
import spacy
from transformers import pipeline

class ApiServer:
    def __init__(self):
        self.app = Flask(__name__)
        self.nlp = spacy.load("vikamigo_data/model")  # Load non-zero-shot model
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    def predict_intent(self, text, candidate_labels):
        return self.classifier(text, candidate_labels)

    def predict_entities(self, text):
        doc = self.nlp(text)
        return [{'entity': ent.text, 'label': ent.label_} for ent in doc.ents]

    def start_server(self):
        @self.app.route('/predict', methods=['POST'])
        def predict():
            data = request.get_json()
            text = data['text']
            candidate_labels = ['Product Inquiry', 'Order Inquiry', 'Technical Support', 'Refund Inquiry', 'Catalog Inquiry', 'Return Request']
            
            intent_result = self.predict_intent(text, candidate_labels)
            entity_result = self.predict_entities(text)
            
            return jsonify({
                'intent': intent_result['labels'][0],  # Get the highest probability label
                'entities': entity_result
            })

        self.app.run(debug=True, host='0.0.0.0', port=5000)
