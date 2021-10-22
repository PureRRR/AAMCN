import time

from sklearn.cluster import SpectralClustering

from metrics import compute_and_print_scores


def specC(data, num_class):
    clustering = SpectralClustering(n_clusters=num_class, assign_labels='kmeans').fit(data)
    return clustering.labels_

def testSpec(data, num_class, num_view):
    for batch in data:
        data = batch
    gt = data['label'].numpy().tolist()
    for j in range(num_view):
        print('view ' + str(j))
        t = time.perf_counter()
        label = specC(data[str(j + 1)], num_class)
        print(len(label))
        print(time.perf_counter()-t)
        compute_and_print_scores(label, gt, mode='test')