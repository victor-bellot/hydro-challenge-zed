# %% Machine Learning Class - K-means Segmentation

# package
import numpy as np
import matplotlib.pyplot as plt

from preprocessing import preprocessing
from unsupervisedTraining import unsupervisedTraining
from selectFeatureVectors import selectFeatureVectors

from displayFeatures2d import displayFeatures2d
from displayImageLabel import displayImageLabel

plt.ioff()  # to see figure before input

training_image_name = '../sunny.jpg'
training_features, training_image = preprocessing(training_image_name, show_image=False, show_feature_analysis=False)

k_means_model = unsupervisedTraining(training_features, method='gmm')

labelLearn = k_means_model.predict(training_features)
displayFeatures2d(training_features, group=labelLearn)

features, nb_px, _ = selectFeatureVectors(training_image)

labels = k_means_model.predict(features)

_, counts = np.unique(labels, return_counts=True)
print(counts)

displayImageLabel(labels, training_image)
