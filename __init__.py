import os
import cv2
import glob
import time
import numpy as np
import tensorflow as tf

from imutils.video import WebcamVideoStream
from scipy.spatial.distance import cdist

from .models import facenet
from .detect_utils import detect_and_align

def get_threshold(embs, labels, distance='cosine'):
	dist_matrix = cdist(embs, embs, distance)
	sigmas = []

	for i, label in enumerate(labels):
		dist = dist_matrix[i]
		dist = dist[np.where(labels == label)]
		dist = 1 - dist 
		# print(dist)
		
		sigma = dist[np.argmin(dist)]
		sigmas.append(sigma)

	return sigmas

class FaceRecognizer(object):
	def __init__(self, registration_folder=None):
		global facenet
		base_path = os.path.dirname(os.path.realpath(__file__))
		weights_path = os.path.join(base_path, 'model_94k_faces_glintasia_without_norm.hdf5')

		print('[INFO] Loading model ... ')
		facenet.load_weights(weights_path)
		self.model = tf.keras.models.Model(inputs=facenet.inputs[0], outputs=facenet.get_layer('emb_output').output)

		if(registration_folder is None):
			self.registration_folder = os.path.join(base_path, 'identities')
		else:
			self.registration_folder = os.path.join(base_path, registration_folder)

		# self.model = facenet 
		self.base_path = os.path.dirname(os.path.realpath(__file__))
		self.embeddings = np.array([])
		self.labels = np.array([])

		print('[INFO] Loading identities ... ')
		for i, idx in enumerate(glob.glob(self.registration_folder + '/*')):
			label = os.path.basename(idx)

			npy_path = os.path.join(idx, '%s.npy' % label)
			embeddings = np.load(npy_path)
			labels = np.full(embeddings.shape[0], label)

			if(i == 0):
				self.embeddings = embeddings
			else:
				self.embeddings = np.concatenate((self.embeddings, embeddings))
			
			self.labels = np.concatenate((self.labels, labels))

		if(len(np.unique(self.labels)) >= 2):
			self.sigmas = get_threshold(self.embeddings, self.labels)

	def _face_preprocessing(self, image, size=(170, 170)):
		image = cv2.resize(image, size)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = (image - 127.5) / 127.5

		return image

	def _is_blur(self, image, threshold=160):
		return cv2.Laplacian(image, cv2.CV_64F).var() < threshold

	def _is_bad_lighting(self, image):
		bright_thres = 0.5
		dark_thres = 0.3
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		dark_part = cv2.inRange(gray, 0, 60)
		bright_part = cv2.inRange(gray, 200, 255)
		# use histogram
		# dark_pixel = np.sum(hist[:30])
		# bright_pixel = np.sum(hist[220:256])
		total_pixel = np.size(gray)
		dark_pixel = np.sum(dark_part > 0)
		bright_pixel = np.sum(bright_part > 0)

		bad_lighting = False
		if dark_pixel/total_pixel > bright_thres:
			bad_lighting = True
		if bright_pixel/total_pixel > dark_thres:
			bad_lighting = True

		return bad_lighting

	def recognize(self, face):
		identity = 'Unknown'
		### Get embeddings from face ###
		embedding = self.model.predict(np.array([face]))
		embedding = embedding / np.linalg.norm(embedding, axis=1).reshape(-1, 1)

		### Get the distance matrix ###
		dist_mat = 1 - cdist(embedding, self.embeddings, 'cosine')
		dist_mat = dist_mat[0]
		best_match = np.argmax(dist_mat)

		if(dist_mat[best_match] >= self.sigmas[best_match] + 0.02):
			identity = self.labels[best_match]

		return identity

	def start_standalone_app(self, video=None):
		videoSrc = 2
		if(video is not None):
			videoSrc = video 

		vs = WebcamVideoStream(src=videoSrc).start()

		while(True):
			frame = vs.read() 
			#if(video is None):
			#	frame = cv2.flip(frame, flipCode=1)
			#	frame = cv2.flip(frame, flipCode=0)

			faces, locations = detect_and_align(frame)
			for face, location in zip(faces, locations):
				x1, y1, x2, y2 = location 
				cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 1)

				### Check image quality ###
				blur_ = self._is_blur(face)
				bad_light_ = self._is_bad_lighting(face)

				label = None
				if(blur_):
					label = 'BLURRY'
				elif(bad_light_):
					label = 'BAD_LIGHTING'
				else:
					label = self.recognize(self._face_preprocessing(face))

				color = (0,255,0) if (blur_ != True and bad_light_ != True) else (0,0,255) 

				cv2.putText(frame, label, (x1,y1), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 2)

			cv2.imshow('Frame', frame)
			key = cv2.waitKey(1)
			if(key == ord('q')):
				break 

		vs.stop()
		cv2.destroyAllWindows()

	def register(self, name, video=None):
		videoSrc = 2
		if(video is not None):
			videoSrc = video 

		print('[INFO] Starting registration ...')
		vid = cv2.VideoCapture(videoSrc)
		start = time.time()
		num_images = 0
		face_images = []
		id_dir = os.path.join(self.registration_folder, name)
		img_dir = os.path.join(id_dir, 'imgs')

		if(not os.path.exists(id_dir)):
			print('[INFO] Creating ID directory ... ')
			os.mkdir(id_dir)

		if(not os.path.exists(img_dir)):
			print('[INFO] Creating images directory ... ')
			os.mkdir(img_dir)

		while(True):
			ret, frame = vid.read()
			#if(video is None):
			#	frame = cv2.flip(frame, flipCode=1)
			#	frame = cv2.flip(frame, flipCode=0)

			### Detection and alignment ###
			faces, locations = detect_and_align(frame)
			
			for face, location in zip(faces, locations):
				cv2.imshow('Face (Aligned)', face)
				x1, y1, x2, y2 = location 

				blur_ = self._is_blur(face)
				bad_light_ = self._is_bad_lighting(face)

				color = (0,255,0) if (blur_ != True) else (0,0,255) 
				blur_condition = 'Good' if (blur_ != True) else 'Not good'
				cv2.putText(frame, 'Blurriness : %s' % blur_condition, (20,20), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 2)
				
				color = (0,255,0) if (bad_light_ != True) else (0,0,255) 
				light_condition = 'Good' if (bad_light_ != True) else 'Not good'
				cv2.putText(frame, 'Lighting condition : %s' % light_condition, (20,40), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 2)

				cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

				if(not blur_ and not bad_light_):
					num_images += 1
					filename   = '%s_%d.jpg' % (name, num_images)
					abs_filename = os.path.join(img_dir, filename)

					cv2.imwrite(abs_filename, face)
					face_images.append(self._face_preprocessing(face))
				else:
					print('[INFO] Skipping bad image ... ')

			cv2.imshow('Frame', frame)
			key = cv2.waitKey(1)
			if(key == ord('q') or time.time() - start >= 5):
				print('[INFO] Stopping registration ... ')
				break

		if(len(face_images) >= 20):
			print('[INFO] Number of images collected : ', len(face_images))
			print('[INFO] Generating embeddings ... ')
			embeddings = self.model.predict(np.array(face_images))
			embeddings = embeddings / np.linalg.norm(embeddings, axis=1).reshape(-1, 1) # normalize

			with open(os.path.join(id_dir, '%s.npy' % name), 'wb') as f:
				np.save(f, embeddings)
				print('[INFO] Saved normalized embeddings to %s' % os.path.join(id_dir, '%s.npy' % name))
		else:
			print('[INFO] Not enough frames registered ... ')

		print('[INFO] Reinitializing ... ')
		self.__init__()
		vid.release()
		cv2.destroyAllWindows()
