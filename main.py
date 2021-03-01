import pandas as pd
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
# Utilities
from time import time
from zipfile import ZipFile
import os, sys, itertools, re
import warnings, pickle, string
from flask import Flask, request, render_template

import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer


app = Flask(__name__)

def generate_summary(context):
    model_name = 'google/pegasus-xsum'
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name)
    batch = tokenizer.prepare_seq2seq_batch(src_texts='context', truncation=True, padding='max-length', return_tensors="pt")
    translated = model.generate(**batch)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return tgt_text



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/demon', methods=['POST', 'GET'])
def submit():
    if request.method == 'POST':
        cont = request.form['context']
        pred= generate_summary([cont])
        return render_template('index.html', data=pred)


if __name__ == "__main__":
    # app.run(host='127.0.0.1', port=8001, debug=True)
    app.run(debug=True)  # running the app
