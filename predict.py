import numpy as np
import pickle

from flask import Flask, jsonify, request
import requests

new_model = 'model.bin'
dv_file = 'dv.bin'

# Load the model, dv
with open(new_model, 'rb') as f_in:
    model = pickle.load(f_in)

with open(dv_file, 'rb') as f_in_dv:
    dv = pickle.load(f_in_dv)

app=Flask('price')

@app.route('/price', methods=['POST'])
def predict():
    laptop = request.get_json()

    X = dv.transform([laptop])
    price = model.predict(X)
    final_price = np.expm1(price[0])
    
    result = {
        'laptop price': final_price
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)

