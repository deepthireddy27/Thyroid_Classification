from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
from collections import OrderedDict
import torch
from torch import nn
from torchvision import transforms, models, datasets
from PIL import Image
import io
import json
# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
import sqlite3

app = Flask(__name__)

    
@app.route("/")
def index():
    return render_template("index.html")

@app.route('/logon')
def logon():
	return render_template('signup.html')

@app.route('/login')
def login():
	return render_template('signin.html')

@app.route("/signup")
def signup():

    username = request.args.get('user','')
    name = request.args.get('name','')
    email = request.args.get('email','')
    number = request.args.get('mobile','')
    password = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("insert into `info` (`user`,`email`, `password`,`mobile`,`name`) VALUES (?, ?, ?, ?, ?)",(username,email,password,number,name))
    con.commit()
    con.close()
    return render_template("signin.html")

@app.route("/signin")
def signin():

    mail1 = request.args.get('user','')
    password1 = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("select `user`, `password` from info where `user` = ? AND `password` = ?",(mail1,password1,))
    data = cur.fetchone()

    if data == None:
        return render_template("signin.html")    

    elif mail1 == 'admin' and password1 == 'admin':
        return render_template("home.html")

    elif mail1 == str(data[0]) and password1 == str(data[1]):
        return render_template("home.html")
    else:
        return render_template("signup.html")

@app.route('/home')
def home():
	return render_template('home.html')

@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method == 'POST':

        # get uploaded image
        file = request.files['image']
        if not file:
            return render_template('home.html', label="No file uploaded")


        img = Image.open(file)
        img = transforms.functional.resize(img, 250)
        img = transforms.functional.five_crop(img, 224)

        f = lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in img])
        feature = f(img)

        k = lambda norm: torch.stack([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop) for crop in feature])
        features = k(feature)
        
        # feed through network
        output = model(features)
        # take average of five crops
        output = output.mean(0)


        # output = model(image)

        output = output.numpy().ravel()
        labels = thresh_sort(output,0.5)
        

        if len(labels) == 0 :
            label = " There are no pascal voc categories in this picture "
            # category = cat_to_name[str(np.argmax(output))]
            # label = " There doesnt seem to be any pascal voc categories in this picture, but if I had to guess it looks like a " + category


        else :
            label_array = [ cat_to_name[str(i)] for i in labels]
            label_array = label_array[0]
           
            label = "Predictions: " + ", ".join(label_array )
        
        return render_template('home.html', label=label)

def thresh_sort(x, thresh):
    idx, = np.where(x > thresh)
    return idx[np.argsort(x[idx])]

def init_model():
    np.random.seed(2019)
    torch.manual_seed(2019)
    resnet = models.resnet18(pretrained=True)
    num_ftrs = resnet.fc.in_features
    resnet.fc = torch.nn.Linear(num_ftrs, 3)
    resnet.load_state_dict(torch.load('resnet18.pth', map_location='cpu'))

    for param in resnet.parameters():
        param.requires_grad = False
    resnet.eval()
    return resnet

@app.route('/graph')
def graph():
	return render_template('graph.html')

@app.route('/notebook')
def notebook():
	return render_template('notebook1.html')

if __name__ == '__main__':
    # initialize model
    model = init_model()
    # initialize labels
    
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    # start app
    app.run(debug=True)