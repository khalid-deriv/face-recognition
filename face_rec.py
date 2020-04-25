import face_recognition
import time
import cv2
import numpy as np

size = 4

picture_of_me = cv2.imread("khalid_ID.jpg")
picture_of_me = cv2.cvtColor(picture_of_me, cv2.COLOR_BGR2RGB)
my_face_encoding = face_recognition.face_encodings(picture_of_me)[0]

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
			results = face_recognition.compare_faces([my_face_encoding], unknown_face_encoding)[0]
		except:
			results = False

		if results == True:
			print("It's a picture of me!")
			text = 'Khalid'
		else:
			print("It's not a picture of me!")
			text = 'unknown'

		font = cv2.FONT_HERSHEY_TRIPLEX
		cv2.putText(im, text,(x+w,y), font, 1, (0,0,255), 2)

	# Show the image
	cv2.imshow('Capture', im)
	key = cv2.waitKey(10)

	if key == 27:
		exit()
