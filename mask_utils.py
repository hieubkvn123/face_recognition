import os
import cv2
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input 

class MaskDetector(object):
	def __init__(self, model_file='mask_detector.model'):
		self.base_path = os.path.dirname(os.path.realpath(__file__))
		self.model_file = os.path.join(self.base_path, model_file) 
		self.mask_detector = load_model(self.model_file)

	def _preprocess(self, image):
		'''
			image : ndarray, bgr image of the cropped face
		'''
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = cv2.resize(image, (224, 224))
		image = img_to_array(image)
		image = preprocess_input(image)

		return image 

	def predict(self, image):
		image = self._preprocess(image)
		label = self.mask_detector.predict(np.array([image]))
		label = np.argmax(label[0])
		if(label == 1):
			label = 'No Mask'
		else:
			label = 'Mask'

		return label