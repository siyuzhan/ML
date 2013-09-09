import kNN, matplotlib.pyplot
import numpy as np

def complete_face(knn, neighbors):
    arr = 0
    sim = 0
    for i in neighbors:
        arr += knn.training_data[i[0]] * i[1]
        sim += i[1]
    return arr / sim

def main(train_path="PS1_data/faces.train", test_path="PS1_data/faces.test", 
         k=1):
    knn = kNN.kNN(k=k, fun=kNN.inverse_euclidean_distance)
    knn.train(train_path)
    knn.load_test_file(test_path)
    for i in knn.test_data:
        neighbors = knn.test(i)
        x = complete_face(knn,neighbors)
        matplotlib.pyplot.gray()
        matplotlib.pyplot.imshow(x.reshape((64,64)))
        matplotlib.pyplot.show()
    
