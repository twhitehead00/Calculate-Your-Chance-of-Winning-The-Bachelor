from flask import Flask, request, jsonify, Response
import model_export as model
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# define an endpoint for making predictions
@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == "OPTIONS":
        return Response(status=200)
    
    input_data = request.json
    processed_data = model.preprocess(input_data)
    # make predictions with the model    
    prediction = model.predict(processed_data)
    return jsonify(int(prediction))