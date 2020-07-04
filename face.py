import cv2
import numpy as np
import face_recognition
from skimage import io
from io import BytesIO
import os
import time
import datetime
import glob
from ftplib import FTP
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
		'user_id': person_id,
		'time': int(time.time())
	})
	print("Door Alarm!")

def get_encodings_list(encodings):
	ids = []
	encodings_list = []
	for itm in encodings:
		ids.append(itm.id)
		itm = itm.to_dict()
		encodings_list.append(itm['face'])
	return ids, encodings_list

face_ref = firestore_db.collection(u'face_encoding')

# connect to FTP server
ftp = FTP(host='ftp.fever-tracker.a2hosted.com', user='face@fever-tracker.a2hosted.com', passwd='EE1yKeY^1Y@K')

ftp.cwd('/faces')

# get all the file entries and reorder them by time modified
entries = list(ftp.mlsd())
entries.sort(key = lambda entry: entry[1]['modify'], reverse = True)
# get the first element in the list, with the first attribute (name)
latest_name = entries[3][0]

# get the file data in bytes
r = BytesIO()
ftp.retrbinary('RETR ' + latest_name, r.write)
value = r.getvalue()

# convert from bytes to nparray image
latest_file = cv2.imdecode(np.frombuffer(value, np.uint8), -1)

files_count = len(entries)
previous_files_count = 0
# Read images
while True:
	if files_count > previous_files_count and latest_name[-3:] == 'jpg':
		time.sleep(1)

		# # Convert to RGB
		picture = cv2.cvtColor(latest_file, cv2.COLOR_BGR2RGB)

		try:
			# # Get encoding
			my_face_encoding = face_recognition.face_encodings(picture)[0]

			ids, face_encodings = get_encodings_list(face_ref.stream())
			# Compare the faces
			results = face_recognition.compare_faces(face_encodings, my_face_encoding)
		except Exception as e:
			print(e)
			results = False
		if results:
			for i in range(len(results)):
				if results[i]: detected(ids[i])

		previous_files_count = files_count
	
	entries = list(ftp.mlsd())
	entries.sort(key = lambda entry: entry[1]['modify'], reverse = True)
	latest_name = entries[3][0]

	r = BytesIO()
	ftp.retrbinary('RETR ' + latest_name, r.write)

	value = r.getvalue()
	latest_file = cv2.imdecode(np.frombuffer(value, np.uint8), -1)
	files_count = len(entries)
	print(files_count)
