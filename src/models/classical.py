import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def get_classical_models(seed, n_classes=None):
    """
    Returns a dict of freshly instantiated classical classifiers.

    Parameters
    ----------
    seed      : int  — random seed for SVM
    n_classes : int  — if given, LDA is initialised with uniform priors (1/n_classes)
    """
    lda = (LinearDiscriminantAnalysis(priors=np.ones(n_classes) / n_classes)
           if n_classes is not None else LinearDiscriminantAnalysis())
    return {
        'LDA':     lda,
        'KNN-3':   KNeighborsClassifier(n_neighbors=3, weights='distance'),
        'SVM-RBF': SVC(kernel='rbf', C=1.0, gamma='scale', probability=True,
                       random_state=seed),
    }
