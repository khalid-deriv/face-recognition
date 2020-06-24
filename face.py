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

# Initialize Firestore and realtime DB
cred = credentials.Certificate('key.json')
default_app = initialize_app(cred, {
    'databaseURL': 'https://fevertracker-4bf99.firebaseio.com'
})
firestore_db = firestore.client()
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

def get_encodings_list(encodings):
	ids = []
	encodings_list = []
	for itm in encodings:
		ids.append(itm.id)
		itm = itm.to_dict()
		encodings_list.append(itm['face'])
	return ids, encodings_list

face_ref = firestore_db.collection(u'face_encoding')

list_of_files = glob.glob('faces/*.jpg')
files_count = len(list_of_files)
previous_files_count = 0

# Read images
while True:
	if files_count > previous_files_count:
		time.sleep(1)
		latest_file = max(list_of_files, key=os.path.getctime)

		picture = cv2.imread(latest_file)

		# # Convert to RGB
		picture = cv2.cvtColor(picture, cv2.COLOR_BGR2RGB)

		# # Get encoding
		my_face_encoding = face_recognition.face_encodings(picture)[0]

		ids, face_encodings = get_encodings_list(face_ref.stream())
		try:
			# Compare the faces
			results = face_recognition.compare_faces(face_encodings, my_face_encoding)
		except Exception as e:
			print(e)
			results = False
		for i in range(len(results)):
			if results[i]: detected(ids[i])

		previous_files_count = files_count
	
	list_of_files = glob.glob('faces/*.jpg')
	files_count = len(list_of_files)
	print(files_count)
