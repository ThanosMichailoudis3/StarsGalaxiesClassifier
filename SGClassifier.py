import numpy as np
import matplotlib.pyplot as plt
from path import Path
import cv2 as cv
import os
import h5py
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import plot_confusion_matrix, roc_curve, classification_report, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
import joblib

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#DATA PROCESSING
# Shape feature: Hu Moments
def fd_hu_moments(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    feature = cv.HuMoments(cv.moments(image)).flatten()
    return feature

# Color feature: Color Histogram
def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    # compute the color histogram
    hist = cv.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv.normalize(hist, hist)
    # return the histogram
    return hist.flatten()

#classses
tr_path = Path("/home/thanos/PyProjects/SGC/data/train")
folders = [dir for dir in sorted(os.listdir(tr_path)) if os.path.isdir(tr_path/dir)]

#feature extraction
fixed_size = tuple((128, 128))
bins = 8
features = []
labels =[]


for clss in folders:
    dir = os.path.join(tr_path, clss)

    # get the current training label
    current_label = clss

    # loop over the images in each sub-folder
    for filename in os.listdir(dir):
    # get the image file name
        file = os.path.join(dir, filename)

        # read the image and resize it to a fixed-size
        image = cv.imread(file)
        image = cv.resize(image, fixed_size)
        
        fv_hu_moments = fd_hu_moments(image)
        fv_histogram  = fd_histogram(image) 
        
        feature = np.hstack([fv_histogram, fv_hu_moments])

        # update the list of labels and feature vectors
        labels.append(current_label)
        features.append(feature)

    print("[STATUS] processed folder: {}".format(current_label))

print("Feature Extraction Completed...")


# encode the target labels
targetNames = np.unique(labels)
le = LabelEncoder()
target = le.fit_transform(labels)
print("Training Labels Encoded...")

# scale features in the range (0-1)
scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_features = scaler.fit_transform(features)
print("Feature Vector Normalized...")

# save the feature vector using HDF5
h5_data = '/home/thanos/PyProjects/SGC/data/data128.h5'
h5_labels = '/home/thanos/PyProjects/SGC/data/labels.h5'

h5f_data = h5py.File(h5_data, 'w')
h5f_data.create_dataset('dataset_1', data = np.array(rescaled_features))

h5f_label = h5py.File(h5_labels, 'w')
h5f_label.create_dataset('dataset_1', data = np.array(target))

h5f_data.close()
h5f_label.close()

print("Features Saved..")

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#MODEL TRAINING
seed = 9

# import the feature vector and training labels
h5f_data  = h5py.File(h5_data, 'r')
h5f_label = h5py.File(h5_labels, 'r')

features_string = h5f_data['dataset_1']
labels_string   = h5f_label['dataset_1']

x_train = np.array(features_string)
y_train = np.array(labels_string)

h5f_data.close()
h5f_label.close()

# verify the shape of the feature vector and labels
print("Features Shape: {}".format(x_train.shape))
print("Labels Shape: {}".format(y_train.shape))

#Grid Search
#Pipeline creation for classifier search
class DummyEstimator(BaseEstimator):
    def fit(self): pass
    def score(self): pass

pln = Pipeline([('clf', DummyEstimator())]) 

prms = [{'clf': [RandomForestClassifier(random_state = seed)],
            'clf__n_estimators': np.array(range(100, 450, 50)),
           'clf__criterion': ['gini', 'entropy']},

           {'clf': [SVC(probability = True, random_state = seed)],
            'clf__kernel': ('linear','poly', 'rbf'), 
            'clf__C': np.logspace(0, 4, 10),
            'clf__degree': np.array(range(2, 6))}]

gs = GridSearchCV(pln, prms, cv = 5, n_jobs = -1, scoring = 'accuracy', verbose = 1, 
                                                                 return_train_score = True)
gs.fit(x_train, y_train)    
print("GridSearch Complete...")
print(gs.best_params_)


#Training Accuracy
scores = cross_val_score(gs, x_train, y_train, cv = 5)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


# save the model to disk
b_est = gs.best_estimator_
joblib.dump(b_est, 'best_model.pkl')



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#TESTING

#testing dataset creation
vl_path = Path("/home/thanos/PyProjects/SGC/data/validation")
h5_data = '/home/thanos/PyProjects/SGC/data/val_data128.h5'
h5_labels = '/home/thanos/PyProjects/SGC/data/val_labels.h5'
folders = [dir for dir in sorted(os.listdir(vl_path)) if os.path.isdir(vl_path/dir)]
features = []
labels =[]

# loop through the test images
for clss in folders:
    dir = os.path.join(vl_path, clss)

    # get the current training label
    current_label = clss

    # loop over the images in each sub-folder
    for filename in os.listdir(dir):
    # get the image file name
        file = os.path.join(dir, filename)

        # read the image and resize it to a fixed-size
        image = cv.imread(file)
        image = cv.resize(image, fixed_size)
        
        #feature extraction
        fv_hu_moments = fd_hu_moments(image)
        fv_histogram  = fd_histogram(image) 
        # maybe add hog
        
        feature = np.hstack([fv_histogram, fv_hu_moments])

        # update the list of labels and feature vectors
        labels.append(current_label)
        features.append(feature)

    print("Processed Folder: {}".format(current_label))

print("Feature Extraction Complete...")

# get the overall feature vector size
print("Feature Vector Size {}".format(np.array(features).shape))

# get the overall training label size
print("Testing Labels {}".format(np.array(labels).shape))

# encode the target labels
testNames = np.unique(labels)
le = LabelEncoder()
target = le.fit_transform(labels)
print("Testing Labels Encoded...")

# scale features in the range (0-1)
scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_features = scaler.fit_transform(features)
print("Feature Vector Normalized...")

print("Target Labels: {}".format(target))
print("Target Labels Shape: {}".format(target.shape))

# save the feature vector using HDF5
h5f_data = h5py.File(h5_data, 'w')
h5f_data.create_dataset('dataset_2', data = np.array(rescaled_features))

h5f_label = h5py.File(h5_labels, 'w')
h5f_label.create_dataset('dataset_2', data = np.array(target))

h5f_data.close()
h5f_label.close()


# import the feature vector and trained labels
h5f_data  = h5py.File(h5_data, 'r')
h5f_label = h5py.File(h5_labels, 'r')

features_string = h5f_data['dataset_2']
labels_string   = h5f_label['dataset_2']

x_val = np.array(features_string)
y_val = np.array(labels_string)
print('Validation Dataset Created')

h5f_data.close()
h5f_label.close()

sc = b_est.score(x_val, y_val)
print("Validation Score:", sc*100,'%')

y_pred = b_est.predict(x_val)
print(classification_report(y_val, y_pred, target_names = folders, digits=3))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#Plots and Graphs

#Confusion Matrix
titles_options = [("Confusion matrix", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(b_est, x_val, y_val,
                                 display_labels = folders,
                                 cmap = plt.cm.Blues,
                                 normalize = normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.rc('font', size = 15)
plt.rc('axes', titlesize = 20)
plt.rc('figure', titlesize = 22)
plt.show()

#ROC Curve
probs = b_est.predict_proba(x_val)
preds = probs[:,1]
ns_probs = [0 for _ in range(len(y_val))]
ns_fpr, ns_tpr, _ = roc_curve(y_val, ns_probs)
fpr, tpr, threshold = roc_curve(y_val, preds)
roc_auc = auc(fpr, tpr)
plt.title('ROC Curve')
plt.plot(ns_fpr, ns_tpr, 'c--', label='No Skill')
plt.plot(fpr, tpr, 'b', label = 'SVM AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'c--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


































