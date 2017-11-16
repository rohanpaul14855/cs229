from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np


def pick_rand_centroids(A, k):
    return np.array(A[np.random.randint(low=0, high=A.shape[0], size=k)], dtype=float)


def color_points(A, centroids):
    """
    Assign each point to a centroid
    :param A: List of points from image
    :param centroids: List of k centroids
    :return: C, list of centroid assignments for each point in A
    """
    C = np.zeros(A.shape[0], dtype=int)
    assert(sum(C < 0) == 0)
    for i in range(A.shape[0]):
        centroid_idx = np.argmin(np.sum((centroids - A[i])**2, axis=1))
        C[i] = centroid_idx
    return C


def calc_centroids(A, C, centroids):
    """
    calculate new centroids based on colored points
    :param A: List of points from image
    :param C: list of centroid assignments for each point in A
    :param centroids: list of previous centroids
    :return: updated centroids 
    """
    updated_centroids = [np.array([0., 0., 0.]) for _ in range(len(centroids))]
    counts = np.zeros(len(centroids), dtype=int)
    for i in range(len(A)):
        updated_centroids[C[i]] += A[i]
        counts[C[i]] += 1
    updated_centroids = [i/j for i, j in zip(updated_centroids, counts)]
    return np.array(updated_centroids, dtype=float)


def main(A, k):
    """
    run the k-means algorithm and compress large image
    :param A: List of points from image
    :param k: number of clusters
    """
    centroids = pick_rand_centroids(A, k)
    iter = 1
    while True:
        print('{} iterations'.format(iter))
        iter += 1
        C = color_points(A, centroids)
        prev_centroids = centroids
        centroids = calc_centroids(A, C, prev_centroids)
        if (np.linalg.norm(centroids - prev_centroids))**2 < 1e-5:
            break


    #reduce number of colors by assigning all points to closest cluster

    A_large = imread('mandrill-large.tiff')
    plt.subplots(1, 2)
    plt.subplot(1, 2, 1)
    plt.title('Uncompressed')
    plt.imshow(A_large)
    A_large = A_large.reshape(-1, 3).astype(np.float64)
    dup = np.zeros(A_large.shape)
    for i in range(A_large.shape[0]):
        c_idx = np.argmin(np.sum((centroids - A_large[i])**2, axis=1))
        dup[i] = centroids[c_idx]
    plt.subplot(1, 2, 2)
    plt.title('Compressed')
    plt.imshow(dup.reshape(512, 512, 3).astype(np.uint8))
    plt.show()

if __name__ == '__main__':
    np.random.seed(0)
    A = imread('mandrill-small.tiff').astype(np.float64)
    A = A.reshape(-1, 3)
    main(A, 16)

