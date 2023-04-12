from flask import Flask, request, jsonify, Response
import model_export as model
from flask_cors import CORS
import json


app = Flask(__name__)
CORS(app)

# define an endpoint for making predictions
@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == "OPTIONS":
        return Response(status=200)
    
    input_data = request.json
    # input_data['fir?'] = input_data.pop('FIR')
    # input_data.pop('name')
    processed_data = model.preprocess(input_data)
    # make predictions with the model    
    prediction = model.predict(processed_data)
    return jsonify(int(prediction))


# if __name__ == "__main__":
#     input_data = {
#         "age": 22,
#         "hometown": "NE",
#         "season": 12,
#         "race": "White",
#         "1-on-1_week": "3",
#         "joke_entrance": "No",
#         "fir?": "Yes",
#         "job_category": "Corporate",
#         "note": 0
#     }

#     processed_data = model.preprocess(input_data)
#     # make predictions with the model
#     prediction = model.predict(processed_data)
#     print(prediction)