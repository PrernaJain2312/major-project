import pandas as pd
import numpy as np
from flask import Flask, request, render_template, jsonify
import pickle
from fastai import *
from fastai.text import *
from fastai.core import *


# load model
path = "/Users/jain/Desktop/deploy model/"
model= load_learner(path,"model.pkl")




# app
app = Flask(__name__)

# routes
@app.route('/', methods=['GET'])

def predict():
    # get data
    data = request.args.get('tweet')
    

    # predictions
    result = model.predict(data)


    # send back to browser
    output = {'result': int(result[0])}

    # return data
    return jsonify(output)


if __name__ == '__main__':
    app.run(port = 2300, debug=True)