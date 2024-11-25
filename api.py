from modules.api_server import ApiServer

def run():
    api_server = ApiServer()
    api_server.start_server()

if __name__ == '__main__':
    run()


#  from flask import Flask, jsonify, request, send_from_directory
# from modules import  IntentModel

# app = Flask(__name__)

# @app.route('/')
# def home():
#     return send_from_directory('view', 'index.html')

# @app.route('/<path:path>')
# def static_files(path):
#     return send_from_directory('view', path)

# @app.route("/predict", methods=["POST"])
# def predict():
#     data = request.json
#     query = data.get("query", "")
#     model, vectorizer = IntentModel.load("model.pkl", "vectorizer.pkl")
#     query_vec = vectorizer.transform([query])
#     intent = model.predict(query_vec)[0]
#     return jsonify({"intent": intent})

# if __name__ == "__main__":
#     app.run(debug=True, port=5000)

