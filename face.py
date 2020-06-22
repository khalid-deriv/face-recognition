import cv2
import numpy as np
import face_recognition
from skimage import io
import os
import time
import datetime
import glob
from firebase_admin import credentials, firestore, storage, initialize_app, db

size = 4

# Initialize Firestore DB
cred = credentials.Certificate('key.json')
default_app = initialize_app(cred, {
    'databaseURL': 'https://fevertracker-4bf99.firebaseio.com'
})
# db = firestore.client()
db_ref = db.reference('door/alarm')
notify_db = db.reference('notifications')

# Initialize firestore
default_storage = initialize_app(cred, {
    'storageBucket': 'fevertracker-4bf99.appspot.com'
}, 'storage_DB')
bucket = storage.bucket(app=default_storage)

def detected(person_id):
	door_status = db_ref.get()
	if (door_status == "0"):
		db_ref.set("1")
	notify_db.push({
		'type': 'blacklist detected',
		'user_id': person_id
	})
	print("Door opened!")

# Read images
while True:
	list_of_files = glob.glob('*.jpg')
	latest_file = max(list_of_files, key=os.path.getctime)

	picture = cv2.imread(latest_file)

	# # Convert to RGB
	picture = cv2.cvtColor(picture, cv2.COLOR_BGR2RGB)

	# # Get encoding
	my_face_encoding = face_recognition.face_encodings(picture)[0]

	blobs = bucket.list_blobs()

	# time_taken = 0
	for blob in blobs:
		url = blob.generate_signed_url(datetime.timedelta(seconds=300), method='GET')
		img = io.imread(url)

		# resize image
		dim = (640, 640)
		resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
		img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

		try:
			# get the encoding of the unknoown face
			unknown_face_encoding = face_recognition.face_encodings(img)[0]

			# Compare the faces
			results = face_recognition.compare_faces([my_face_encoding], unknown_face_encoding)[0]
		except Exception as e:
			print(e)
			results = False
		if results : detected(blob.name[-2:])
		print(results, blob.name[-2:])
