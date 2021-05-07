#!/usr/bin/env python3
from flask import Flask, render_template, request, Markup, Response, url_for, redirect, abort, globals, \
    send_from_directory
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import and_
from sqlalchemy import or_
import numpy as np
import pandas as pd
import cv2
import joblib, os
import datetime
import warnings

import os

from werkzeug.utils import secure_filename

warnings.filterwarnings('ignore')
from glob import glob
import joblib
from sklearn.preprocessing import LabelEncoder
import keras
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from collections import deque
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import model_from_json
from pathlib import Path

projectfile = os.path.abspath(os.path.dirname(__file__))
app = Flask(__name__)

database_file = "sqlite:///{}".format(os.path.join(projectfile, "record.db"))
app.config["SQLALCHEMY_DATABASE_URI"] = database_file
app.config[" SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)
UPLOAD_FOLDER = 'uploads'
app.config['SECRET_KEY'] = 'bc3a3cbf29c31d9b32690d356360d075'
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.jpeg']
app.config['UPLOAD_PATH'] = 'uploads'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png'])


class Records(db.Model):
    id = db.Column(db.Integer, unique=True, nullable=False, primary_key=True)
    firstname = db.Column(db.String(500), unique=False, nullable=False)
    lastname = db.Column(db.String(500), unique=False, nullable=False)
    email = db.Column(db.String(500), unique=False, nullable=False)
    image = db.Column(db.String(500), unique=False, nullable=False)
    result = db.Column(db.String(500), unique=False, nullable=False)
    contact = db.Column(db.String(500), unique=False, nullable=False)

db.create_all()


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


# function to predict result
def gen_result(uploaded_image):
    K.clear_session()
    # tf.reset_default_graph()
    class_label = ["COVID-19 CASE", "NON COVID-19 CASE", "Non Relevant"]
    model = load_model('model.h5')
    # class_label = ["NOT COVID-19 CASE", " COVID-19 CASE", "Not Relevant"]
    print(class_label)

    img = image.load_img('static/uploads/' + uploaded_image, target_size=(150, 150, 3))

    test_image = image.img_to_array(img) / 255

    listOfImages = np.expand_dims(test_image, axis=0)

    result = model.predict(listOfImages)

    single_result = result[0]

    print('result[o]: ', result[0])

    mostLikelyClass = round(np.argmax(single_result) // 1)

    class_likelihood = single_result[mostLikelyClass]

    class_label = class_label[mostLikelyClass]

    print('Class:', class_label)
    print("Prediction: Image is a {} - likelihood: {:2f}%".format(class_label, class_likelihood))

    if result[0][0] == 1:
        prediction = ('PREDICTION: COVID-19 Positive likelihood:', round(class_likelihood * 100, 2), '%')

    elif result[0][1] == 1:
        prediction = ('PREDICTION: Non Covid (Healthy) or Uploaded image is irrelevant! likelihood:',
                      round(class_likelihood * 100, 2), '%')

    elif result[0][2] == 1:
        prediction = ('PREDICTION: Non Covid (Healthy) or Uploaded image is irrelevant! likelihood:',
                      round(class_likelihood * 100, 2), '%')

        # full_details = [prediction]

    prediction = class_label

    """allpred=[]
    for each in result:
        allpred.append(each*100)
    print(allpred)"""
    cP = round((result[0][0]) * 100, 2)
    ncP = round((result[0][1]) * 100, 2)
    irP = round((result[0][2]) * 100, 2)
    print('first R:', ncP, 'second R:', irP)
    percent = round((class_likelihood * 100), 2)
    ppercent = percent
    # opercent =

    full_result = ('Covid Possibility:', cP,
                   '%  Non-Covid Possibility:', ncP,
                   '%  Irrelevant-Image Possibility:', irP, '%')
    full_result_str = " ".join(map(str, full_result))

    """full_details = [prediction, 'Possibility:', ppercent,
                    '%  Covid Possibility:', cP,
                    '%  Non-Covid Possibility:', ncP,
                    '%  Irrelevant-Image Possibility:', irP, '%']"""

    full_details = [prediction,
                    """||  Covid Possibility:""", cP,
                    """% || Non-Covid Possibility:""", ncP,
                    """% || Irrelevant-Image Possibility:""", irP, '%']

    full_details_str = " ".join(map(str, full_details))

    # return full_details, prediction
    # return prediction, ppercent
    print("1",full_details_str)
    print("1",full_result_str)
    return full_details_str


"""

    #test_image = image.load_img('/content/drive/MyDrive/FinalYearProject/datasets/CT_NonCOVID/10%2.jpg')
    #test_image= image.load_img('/content/drive/MyDrive/FinalYearProject/datasets/CT_NonCOVID/10%2.jpg', target_size = (150,150,3))

    img = image.load_img('static/uploads/' + uploaded_image, target_size=(150, 150, 3))

    test_image = image.img_to_array(img) / 255
    listOfImages = np.expand_dims(test_image, axis=0)
    percent = model.predict(listOfImages) * 100

    single_result = percent[0]

    mostLikelyClass = round(np.argmax(single_result) // 1)
    class_likelihood = single_result[mostLikelyClass]
    class_label = class_label[mostLikelyClass]
    trial = ("Prediction: Image is a {} \n- likelihood: {:2f}%\n".format(class_label, class_likelihood))
    result = np.argmax(percent)

    if result[0][0] == 1:
        print("COVID")
    elif result[0][1] == 1:
        print("non-COVID")
    elif result[0][2] == 1:
        print("non-relevant")
    prediction = class_label[result]

    print('Class:',class_label)
    #print("Prediction: Image is a {} - likelihood: {:2f}%".format(class_label, class_likelihood))

    full_details = [prediction, percent, trial]"""
"""
    full_details = [prediction, percent, trial]
    return full_details, prediction"""

# function to get patient record
'''def get_record(patient):
    # patient = str(patient).lower()
    # records = pd.read_csv("records.csv")
    # names = records['Names'].values
    # result = 'None'
    # if patient in names:
    #     index = np.where(names == patient)[0][0]
    #     result = records['Record'][index]
    return result'''


# home page
@app.route('/')
@app.route('/home', methods=['GET', 'POST'])
def home():
    # global COUNT
    pred_response = "Upload Image"
    if request.method == 'POST':
        target_img = os.path.join(projectfile, 'static/uploads')
        if not os.path.isdir(target_img):
            os.mkdir(target_img)

        uploaded_file = request.files['image_file']
        if uploaded_file.filename != '':
            filename = uploaded_file.filename
            destination = "/".join([target_img, filename])
            uploaded_file.save(destination)
            qprediction = gen_result(filename)
            print('qprediction', qprediction)
            pred_response = qprediction
            class_0 = pred_response[0]
            class_1 = pred_response[0]
            class_2 = pred_response[0]
            form = request.form
            return render_template('home.html', title='Home', pred_response=pred_response,
                                   filename=filename, image=filename, scroll='quickScan',
                                   class_0=class_0, class_1=class_1)

    # return redirect(url_for('main'))
    form = request.form

    return render_template('home.html', title='Home', pred_response=pred_response)


# about page
@app.route('/about')
def about():
    return render_template('about.html')


# main page
@app.route('/main', methods=['GET', 'POST'])
def main():
    pred_response = "Upload Image"
    patient_id = "Not Found"
    if request.method == 'POST':
        target_img = os.path.join(projectfile, 'static/uploads')
        if not os.path.isdir(target_img):
            os.mkdir(target_img)
        uploaded_file = request.files['image_file']
        filename = uploaded_file.filename

        form = request.form
        first_name = request.form['fname']
        last_name = request.form['lname']
        email = request.form['mail']
        contact = request.form['contact']
        record = Records()
        record.firstname = first_name
        record.lastname = last_name
        record.email = email
        record.contact = contact

        record.image = filename
        destination = "/".join([target_img, record.image])

        uploaded_file.save(destination)
        qprediction = gen_result(filename)
        record.result = qprediction
        db.session.add(record)
        db.session.commit()

        print(id, first_name, last_name, email, contact, pred_response,  record.image)


        pred_response = qprediction
        db.session.add(record)
        db.session.commit()
        class_0 = pred_response[0]
        class_1 = pred_response[0]
        patient_id = record.id
        form = request.form
        return render_template('main.html', title='Main', form=form, pred_response=qprediction, patient_id=patient_id,
                               filename=filename, class_0=class_0, class_1=class_1)
    # return redirect(url_for('main'))
    form = request.form

    return render_template('main.html', title='Main', form=form, pred_response=pred_response, patient_id=patient_id)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_PATH'],
                               filename)


@app.route('/record', methods=['GET', 'POST'], defaults={"page": 1})
@app.route('/record<int:page>', methods=['GET', 'POST'])
def record(page):
    page = page
    pages = 500
    # record_found= Records.query.all()
    record_found = Records.query.order_by(Records.id.asc()).paginate(page, pages, error_out=False)  # desc()
    if request.method == 'POST' and 'tag' in request.form:
        tag = request.form["tag"]
        search = "%{}%".format(tag)
    record_found = Records.query.paginate(page, pages, error_out=False)
    rows = record_found

    return render_template('record.html', title='Records', rows=rows, search1="", search2="d-none")


@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        id = request.form['uid']
        if id != '':
            rows = Records.query.filter_by(id=id).first()
            return render_template('record.html', title='Records', rows=rows, search1='d-none', search2="")
        else:
            return redirect(url_for('record'))

    return redirect(url_for('record'))


@app.route('/delete', methods=['GET', 'POST'])
def delete():
    if request.method == 'POST':
        id = request.form['delid']
        if id != '':
            rows = Records.query.filter_by(id=id).first()
            db.session.delete(rows)
            db.session.commit()
            print("Deleted record for Patient id:", id)
            message = 'Note: Entered Patient record is Deleted!'
            return redirect(url_for("record"))
        else:
            return redirect(url_for('record'))

    return  redirect(url_for('record'))


"""@app.route('/record', methods=['GET', 'POST'])
def record():
    record_found = 'None'
    rows = [[]]
    message = ''

    form = request.form
    print(request.form)
    if request.method == 'POST':
        if "Search" in request.form.values():
            spatient = request.form['uid'] but i have to set the database  so do i have to del this for now comment it 
                 
            if spatient != '':
                con = create_connection(db_path)
                record_found = fetch_record(con, spatient)
                con.close()
                rows = record_found

            elif spatient.isalpha():
                message = 'Note: Please use only whole digits'
            else:
                message = 'Note: Please Enter Patient ID'

        elif "DELETE" in request.form.values():
            del_id = request.form['delid']
            if del_id != '':
                con = create_connection(db_path)
                delete_record(con, del_id)
                message = 'Note: Entered Patient record is Deleted!'
            elif del_id.isalpha():
                message = 'Note: Please use whole digits ONLY!'
            else:
                message = 'Note: Please enter Patient ID in delete section!'

        elif "Show all" in request.form.values():
            patient = request.form['uid']
            con = create_connection(db_path)
            record_found = fetch_allrecord(con, patient)
            con.close()
            rows = record_found

    return render_template('record.html', title='Records', form=form, rows=rows, message=message)"""

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, use_reloader=True, debug=True)
