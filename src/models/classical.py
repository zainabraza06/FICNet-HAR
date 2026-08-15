import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def get_classical_models(seed, stage='s1'):
    priors = np.ones(4)/4 if stage == 's2a' else None
    lda = LinearDiscriminantAnalysis(priors=priors) if priors is not None else LinearDiscriminantAnalysis()
    
    return {
        'LDA': lda,
        'KNN-3': KNeighborsClassifier(n_neighbors=3, weights='distance'),
        'SVM-RBF': SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=seed),
    }
