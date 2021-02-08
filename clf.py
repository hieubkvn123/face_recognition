import os
import pickle
import numpy as np

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class EmbeddingClassifier(object):
	def __init__(self, registration_folder=None):
		super(EmbeddingClassifier, self).__init__()
		self.base_path = os.path.dirname(os.path.realpath(__file__))

		if(registration_folder is None):
			self.registration_folder = os.path.join(self.base_path, 'identities')
		else:
			self.registration_folder = os.path.join(self.base_path, registration_folder)

		self.embeddings = np.array([])
		self.labels = np.array([])

		id_count = 0
		for (dir_, dirs, files) in os.walk(self.registration_folder):
			if(dir_ != self.registration_folder):
				for file_ in files:
					abs_path = os.path.join(dir_, file_)
					if(abs_path.endswith('.npy')):
						embeddings = np.load(abs_path)
						labels = np.full(embeddings.shape[0], abs_path.split('/')[-1].split('.')[0])

						if(id_count == 0):
							self.embeddings = embeddings 
							self.labels = labels
						else:
							self.embeddings = np.concatenate((self.embeddings, embeddings))
							self.labels = np.concatenate((self.labels, labels))

						id_count += 1

		self.embeddings = np.array(self.embeddings)
		self.labels = np.array(self.labels)
		X_train, X_test, Y_train, Y_test = train_test_split(self.embeddings, self.labels, test_size=0.333)

		self.model = SVC(kernel='rbf', probability=True)
		self.model.fit(X_train, Y_train)

		test_pred = self.model.predict(X_test)
		train_pred = self.model.predict(X_train)

		train_acc = accuracy_score(train_pred, Y_train)
		test_acc = accuracy_score(test_pred, Y_test)

		print('[INFO] Train accuracy = %.2f, Test accuracy = %.2f' % (train_acc, test_acc))

		# print(self.embeddings, self.labels)