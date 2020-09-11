from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)
app.config["DEBUG"] = True

FILENAME = 'finalized_model.sav'
CLASSIFIER = None


@app.route('/', methods=['GET'])
def home():
    return "<h1>Welcome to our flower classification API</h1>"

@app.route('/api/v1/iris/test', methods=["POST"])
def test():
    content = request.json
    return jsonify({"data": content})

@app.route('/api/v1/iris/predict', methods=["POST"])
def predict():
    content = request.json

    model_data = [[content['SL'], content['SW'], content['PL'], content['PW']]]

    global CLASSIFIER
    if CLASSIFIER is None:
        CLASSIFIER = pickle.load(open(FILENAME, 'rb'))

    result = CLASSIFIER.predict(model_data)
    return jsonify({"class": int(result[0])})

app.run()