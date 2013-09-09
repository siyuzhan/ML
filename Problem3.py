import kNN, string, numpy

def load_titles():
    f1 = open("PS1_data/books.train.titles", 'r')
    TRAIN_TITLES = [x for x in f1]
    f2 = open("PS1_data/books.test.titles", 'r')
    TEST_TITLES = [x for x in f2]
    f1.close()
    f2.close()
    return TRAIN_TITLES, TEST_TITLES

def partA():
    TRAIN_TITLES, toss = load_titles()
    knn = kNN.kNN(10, kNN.cosine_similarity)
    knn.train("PS1_data/books.train")
    knn.load_test_file("PS1_data/books.train")
    n = TRAIN_TITLES.index(
            'title-Fifty Shades of Grey: Book One of the Fifty Shades Trilogy\n'
            )
    m = TRAIN_TITLES.index('title-Brains: A Zombie Memoir\n')
    x = knn.test_num(n)
    y = knn.test_num(m)
    a = [TRAIN_TITLES[n[0]] for n in x]
    b = [TRAIN_TITLES[n[0]] for n in y]
    print "Fifty Shades of Grey is most similar to:"
    print string.join(a)
    print "Brains: A Zombie Memoir is most similar to:"
    print string.join(b)

def partB():
    knn = kNN.kNN(1, kNN.cosine_similarity)
    knn.train("PS1_data/books.train")
    knn.load_test_file("PS1_data/books.test")
    correct_pred, predict, true = [0 for i in range(5)], [0 for i in range(5)], [0 for i in range(5)]
    genre_centroid = kNN.get_genre_centroid(knn.training_data, knn.training_labels)
    for i in range(len(knn.test_data)):
        sim_max, label = -1, -1
        for j in range(len(genre_centroid)):
            sim = kNN.cosine_similarity(knn.test_data[i], genre_centroid[j])
            if (sim > sim_max):
                sim_max = sim
                label = j
        if (label == int(knn.test_labels[i])):
            correct_pred[label] += 1
        predict[label] += 1
        true[int(numpy.asscalar(knn.test_labels[i]))] +=1
    return correct_pred, predict, true

def partC(k = 5):
    train_titles, test_titles = load_titles()
    knn = kNN.kNN(10, kNN.cosine_similarity)
    knn.train("PS1_data/books.train")
    knn.load_test_file("PS1_data/books.test")
    results = [knn.choose_label(i) for i in knn.test_data]
    return results
