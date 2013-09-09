import PS1, string

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
    knn = PS1.kNN(10, PS1.cosine_similarity)
    knn.train("PS1_data/books.train")
    knn.load_test_file("PS1_data/books.train")
    n = TRAIN_TITLES.index(
            'title-Fifty Shades of Grey: Book One of the Fifty Shades Trilogy\n'
            )
    m = TRAIN_TITLES.index('title-Brains: A Zombie Memoir\n')
    x = knn.test_num(n, True)
    y = knn.test_num(m, True)
    a = [TRAIN_TITLES[n[0]] for n in x]
    b = [TRAIN_TITLES[n[0]] for n in y]
    print "Fifty Shades of Grey is most similar to:"
    print string.join(a)
    print "Brains: A Zombie Memoir is most similar to:"
    print string.join(b)

def partB():
    knn = PS1.kNN(1, PS1.cosine_similarity)
    knn.train("PS1_data/books.train")
    knn.load_test_file("PS1_data/books.test")
    sim_max, label = -1, -1
    correct_pred, predict, true = [], [], []
    genre_centroid = PS1.get_genre_centroid(knn.training_data, knn.training_labels)
    for i in range(len(knn.test_data)):
        for j in range(len(genre_centroid)):
            sim = PS1.cosine_similarity(knn.test_data[i], genre_centroid[j])
            if (sim > sim_max):
                label = j
        if (label == knn.test_labels[i]):
            correct_pred[label] += 1
        predict[label] += 1
        true[knn.test_labels[i]] +=1
    return correct_pred, predict, true

