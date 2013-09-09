import svmlight_loader as svml
import numpy as np

# similarity functions
def inverse_euclidean_distance(v1, v2):
    n = np.linalg.norm(v1 - v2)
    if n == 0:
        return 1.
    else:
        return 1. / n

def cosine_similarity(v1, v2):
    num = np.dot(v1, v2)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    return num / denom

# kNN implementation
class kNN:
    def __init__(self, k=2, fun=inverse_euclidean_distance):
        self.k = k
        self.fun = fun
        self.training_data = None
        self.training_labels = None
        self.test_data = None
        self.test_labels = None

    def train(self, path):
        train_sparse, self.training_labels = svml.load_svmlight_file(path)
        train_dense = train_sparse.todense()
        self.training_data = np.asarray(train_dense)

    def load_test_file(self, path):
        test_sparse, toss = svml.load_svmlight_file(path)
        test_dense = test_sparse.todense()
        self.test_data = np.asarray(test_dense)

    def test(self, vector):
        dist = [self.fun(i, vector) for i in self.training_data]
        knn = [(-1, float("inf")) for i in range(self.k)]
        for i in range(len(dist)):
            for j in knn:
                if not j[1] < dist[i]:
                    knn.append((i, dist[i]))
                    knn.remove(j)
                    break
        return knn

    def test_num(self, i):
        return self.test(self.test_data[i])
