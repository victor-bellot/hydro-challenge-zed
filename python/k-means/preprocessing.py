# %% package
import numpy as np
import matplotlib.pyplot as plt

from selectFeatureVectors import selectFeatureVectors
from displayFeatures2d import displayFeatures2d
from displayFeatures3d import displayFeatures3d


# %% def preprocessing
def preprocessing(training_image_name, show_image=False, show_feature_analysis=False):
    training_image = plt.imread(training_image_name)

    if show_image:
        plt.imshow(training_image)
        plt.title('Training image')
        plt.show()

    training_features, nb_extracted_pixel, depth = selectFeatureVectors(training_image, nbSubSample=1000)

    if show_feature_analysis:
        displayFeatures2d(training_features, group='r')
        # displayFeatures3d(training_features)
        plt.show()

    return training_features, training_image
