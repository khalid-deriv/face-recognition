import time
import datetime
import cv2
import face_recognition
from skimage import io
from firebase_admin import credentials, firestore, storage, initialize_app

picture_of_me = cv2.imread("../hosam.jpg")
picture_of_me = cv2.cvtColor(picture_of_me, cv2.COLOR_BGR2RGB)
my_face_encoding = face_recognition.face_encodings(picture_of_me)[0]

# Initialize Firestore DB
cred = credentials.Certificate('../key.json')
default_app = initialize_app(cred, {
    'storageBucket': 'fevertracker-4bf99.appspot.com'
})
bucket = storage.bucket(app=default_app)

start = time.time()
blobs = bucket.list_blobs()
end = time.time()
time_taken = float(end - start)

count = 0
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
	except Exception:
		results = False

	print(results, blob.name)
	count += 1
# end = time.time()

print("Total time taken:", str(time_taken), "seconds")