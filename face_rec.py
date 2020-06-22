import cv2
import numpy as np
import face_recognition
import time
from firebase_admin import credentials, firestore, initialize_app, db

size = 4

# Initialize Firestore DB
cred = credentials.Certificate('key.json')
default_app = initialize_app(cred, {
    'databaseURL': 'https://fevertracker-4bf99.firebaseio.com'
})
# db = firestore.client()
db_ref = db.reference('door/open')

def open_door():
	door_status = db_ref.get()
	if (door_status == "0"):
		db_ref.set("1")
	print("Door opened!")

# Read images

picture_of_me = cv2.imread("khalid_ID.jpg")
picture_of_youssef = cv2.imread("youssef.jpg")
picture_of_hosam = cv2.imread("hosam.jpg")
# Convert to RGB
picture_of_me = cv2.cvtColor(picture_of_me, cv2.COLOR_BGR2RGB)
picture_of_youssef = cv2.cvtColor(picture_of_youssef, cv2.COLOR_BGR2RGB)
picture_of_hosam = cv2.cvtColor(picture_of_hosam, cv2.COLOR_BGR2RGB)
# Get encoding
my_face_encoding = face_recognition.face_encodings(picture_of_me)[0]
youssef_encoding = face_recognition.face_encodings(picture_of_youssef)[0]
hosam_encoding = face_recognition.face_encodings(picture_of_hosam)[0]

# load the xml HAAR cascade file
classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

webcam = cv2.VideoCapture(0) #Using default WebCam connected to the PC.

while True:
	(rval, im) = webcam.read()
	im=cv2.flip(im,1,0) #Flip to act as a mirror

	# Resize the image to speed up detection
	mini = cv2.resize(im, (int(im.shape[1]/size), int(im.shape[0]/size)))

	# detect MultiScale / faces
	faces = classifier.detectMultiScale(mini)

	text = ''
	status = ''

	if len(faces) > 0:
		# get the first face only
		f = faces[0]
		# Draw rectangles around the face
		(x, y, w, h) = [v * size for v in f] #Scale the shapesize backup
		cv2.rectangle(im, (x,y), (x+w,y+h), (0,255,0), 4)

		#Save just the rectangle faces in SubRecFaces
		sub_face = im[y:y+h, x:x+w]

		try:
			# get the encoding of the unknoown face
			unknown_face_encoding = face_recognition.face_encodings(sub_face)[0]

			# Compare the faces
			results = face_recognition.compare_faces([my_face_encoding, youssef_encoding, hosam_encoding], unknown_face_encoding)
			# results = face_recognition.compare_faces([my_face_encoding, hosam_encoding], unknown_face_encoding)

		except:
			results = [False, False, False]
			print("Error occured")
			# results = [False, False]

		if results[0] == True:
			print("It's a match!")
			text = 'Khalid'
			status = 'Suspected'
			# open_door()
			print("Entry denied!")
		elif results[1] == True:
			print("It's a match!")
			text = 'Youssef'
			status = 'Healthy'
			open_door()
		elif results[1] == True:
			print("It's a match!")
			text = 'Hosam'
			status = 'Healthy'
			open_door()

		else:
			print("It's not a match!")
			text = 'unknown'

		font = cv2.FONT_HERSHEY_TRIPLEX
		cv2.putText(im, text,(x+w+5,y), font, 1, (255,0,0), 2)
		cv2.putText(im, status,(x+w+5,y+30), font, 0.8, (0,0,255), 1)

	# Show the image
	cv2.imshow('Capture', im)
	key = cv2.waitKey(10)

	if key == 27:
		exit()
