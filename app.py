import os
import datetime
from flask import Flask, request, jsonify
from firebase_admin import credentials, firestore, storage, initialize_app

# Initialize Flask App
app = Flask(__name__)

# Initialize Firestore DB
cred = credentials.Certificate('key.json')
default_app = initialize_app(cred, {
    'storageBucket': 'fevertracker-4bf99.appspot.com'
})
bucket = storage.bucket(app=default_app)

blob = bucket.blob('uploads/3')

url = blob.generate_signed_url(datetime.timedelta(seconds=300), method='GET')
# db = firestore.client()
# todo_ref = db.collection('todos')