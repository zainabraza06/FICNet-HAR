from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def get_classical_models(seed):
    return {
        'LDA': LinearDiscriminantAnalysis(),
        'KNN-3': KNeighborsClassifier(n_neighbors=3, weights='distance'),
        'SVM-RBF': SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=seed),
    }
