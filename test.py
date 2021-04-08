import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('TkAgg')

from models import facenet
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--weights-path", type=str, required=True, help="Path to the model weights")
args = vars(parser.parse_args())

base_path = os.path.dirname(os.path.realpath(__file__))
facenet.load_weights(os.path.join(base_path, args['weights_path']))
facenet = tf.keras.models.Model(inputs=facenet.inputs[0], outputs=facenet.get_layer('emb_output').output)

def neutralize_image(img):
    r,g,b = cv2.split(img)
    r = cv2.equalizeHist(r)
    g = cv2.equalizeHist(g)
    b = cv2.equalizeHist(b)

    rgb = cv2.merge((r,g,b))
    return rgb

def preprocessing(img):
    img = cv2.resize(img, (170, 170))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = neutralize_image(img)

    img = (img - 127.5) / 127.5

    ### Standardization ###
    std = np.std(img)
    mu  = np.mean(img)

    img = (img - mu) / std

    return img

images = []
labels = []
datadir = os.path.join(base_path, 'data')
num_files = 0
for (dir_, dirs, files) in os.walk(datadir):
    if(dir_ != datadir):
        label = dir_.split('/')[-1]
        for file_ in files:
            if(file_.endswith('.jpg')):
                abs_path = os.path.join(dir_, file_)
                print('Preprocessing file %s' % abs_path)
                num_files += 1
                img = cv2.imread(abs_path)
                img = preprocessing(img)

                images.append(img)
                labels.append(label)

print(f'[INFO] Number of faces for testing : {num_files} ... ')
images = np.array(images)
labels = np.array(labels)
assert(images.shape[0] == labels.shape[0])

embeddings = facenet.predict(images)
embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1).reshape(-1, 1)
embeddings_pca = PCA(n_components = 3).fit_transform(embeddings_norm)
embeddings_pca_norm = embeddings_pca / np.linalg.norm(embeddings_pca, axis=1).reshape(-1, 1)

fig = plt.figure()
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

for label in np.unique(labels):
    cluster = embeddings_pca_norm[labels == label]
    ax1.scatter(cluster[:,0], cluster[:,1], cluster[:,2])

    cluster = embeddings_pca[labels == label]
    ax2.scatter(cluster[:,0], cluster[:,1], cluster[:,2])

ax1.set_title('Face embeddings (normalized)')
ax2.set_title('Face embeddings (not normalized)')
plt.show()

### Build classifiers and compare ###
print('[INFO] Building classifiers ...')
print('-------------------------------------------------------------------')

X_train, X_test, Y_train, Y_test = train_test_split(embeddings, labels, test_size = 0.33333)
clf_linear_svc = LinearSVC()
clf_linear_svc.fit(X_train, Y_train)
acc_linear_svc = accuracy_score(clf_linear_svc.predict(X_test), Y_test)

clf_rbf_svc = SVC(kernel='rbf', C=1.2, probability=True)
clf_rbf_svc.fit(X_train, Y_train)
acc_rbf_svc = accuracy_score(clf_rbf_svc.predict(X_test), Y_test)

clf_xgb = XGBClassifier()
clf_xgb.fit(X_train, Y_train)
acc_xgb = accuracy_score(clf_xgb.predict(X_test), Y_test)

print(f'[INFO] Accuracy of Linear SVM : {acc_linear_svc} ... ')
print(f'[INFO] Accuracy of RBF SVM : {acc_rbf_svc} ... ')
print(f'[INFO] Accuracy of xgboost classifier : {acc_xgb} ... ')
