# %% package
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


# %% def unsupervisedTraining
def unsupervisedTraining(featLearn, nbCluster=4, method='kmeans'):
    # define class number
    if nbCluster is None:
        answer = input('nombre de classes:')
        nbCluster = int(answer)

    model = None
    if method == 'kmeans':
        model = KMeans(n_clusters=nbCluster,
                       init='k-means++',
                       n_init=10,
                       max_iter=300,
                       tol=1e-4,
                       verbose=0)

    elif method == 'gmm':
        model = GaussianMixture(n_components=nbCluster,
                                covariance_type='full',
                                tol=1e-3,
                                reg_covar=1e-06,
                                max_iter=100,
                                n_init=1,
                                init_params='kmeans',
                                verbose=0,
                                verbose_interval=10)

    model.fit(featLearn)

    return model
