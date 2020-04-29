import cv2
from matplotlib import pyplot
from PIL import Image
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from mtcnn import MTCNN
from keras.models import load_model
from keras_vggface.utils import preprocess_input
# from flask import Flask, request, jsonify
from firebase_admin import credentials, firestore, storage, initialize_app, db


# extract a single face from a given photograph
def extract_face(filename, required_size=(224, 224)):
	# load image from file
	pixels = pyplot.imread(filename)
	# create the detector, using default weights
	detector = MTCNN()
	# detect faces in the image
	results = detector.detect_faces(pixels)
	# extract the bounding box from the first face
	x1, y1, width, height = results[0]['box']
	x2, y2 = x1 + width, y1 + height
	# extract the face
	face = pixels[y1:y2, x1:x2]
	# resize pixels to the model size
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = np.asarray(image)
	return face_array
 
# extract faces and calculate face embeddings for a list of photo files
def get_embeddings(filenames):
	# extract faces
	faces = [extract_face(f) for f in filenames]
	# convert into an array of samples
	samples = np.asarray(faces, 'float32')
	# prepare the face for the model, e.g. center pixels
	samples = preprocess_input(samples, version=2)
	# load the vggface model
	model = load_model('VGGFace.h5') # VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
	# perform prediction
	yhat = model.predict(samples)
	return yhat
 
# determine if a candidate face is a match for a known face
def is_match(known_embedding, candidate_embedding, thresh=0.5):
	# calculate distance between embeddings
	score = cosine(known_embedding, candidate_embedding)
	if score <= thresh:
		return True, score, thresh # print('>face is a Match (%.3f <= %.3f)' % (score, thresh))
	else:
		return False, score, thresh #print('>face is NOT a Match (%.3f > %.3f)' % (score, thresh))

size = 4

# load the model
model = load_model('VGGFace.h5')

# load the xml file
classifier = cv2.CascadeClassifier('../haarcascade_frontalface_alt.xml')

# load image database
image_db = pd.read_csv('image_db.csv')

webcam = cv2.VideoCapture(0) #Using default WebCam connected to the PC.

while True:
	(rval, im) = webcam.read()
	im=cv2.flip(im,1,0) #Flip to act as a mirror

	# Resize the image to speed up detection
	mini = cv2.resize(im, (int(im.shape[1]/size), int(im.shape[0]/size)))

	# detect MultiScale / faces
	faces = classifier.detectMultiScale(mini)
	if len(faces):
		# get the first face only
		f = faces[0]
		# Draw rectangles around the face
		(x, y, w, h) = [v * size for v in f] #Scale the shapesize backup
		cv2.rectangle(im, (x,y), (x+w,y+h), (0,255,0), 4)

	# Show the image
	cv2.imshow('Capture', im)
	key = cv2.waitKey(10)

	if key == 32 and len(faces):
		# get the first face only
		f = faces[0]
		(x, y, w, h) = [v * size for v in f] #Scale the shapesize backup

		#Save just the rectangle faces in SubRecFaces
		sub_face = im[y:y+h, x:x+w]
		sub_face = cv2.resize(sub_face, (224, 224))
		samples = np.asarray(sub_face, 'float32')
		arr4d = np.expand_dims(samples, 0)
		samples_array = preprocess_input(arr4d, version=2)
		embedding_main = model.predict(samples_array)
		embedding_main = embedding_main[0]
		grant = False
		for index, row in image_db.iterrows():
			# get embeddings file filenames
			embeddings = get_embeddings([row['image_path']])

			match, score, thresh = is_match(embedding_main, embeddings[0])
			if match:
				print('>face is a Match with guest %s (%.3f <= %.3f)' % (row['name'], score, thresh))
				grant = True
		if not grant: print('>face is NOT a Match')

	# if Esc key is press then break out of the loop 
	elif key == 27: #The Esc key
		break
	
