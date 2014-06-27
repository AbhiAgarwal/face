'''
- Each principal component has the same length as the original image, thus it can be displayed as an image.
- The Journal of Cognitive Neuroscience referred to these ghostly looking faces as Eigenfaces, that's where the Eigenfaces method got its name from.
'''

import numpy as np
import cv2, os, sys, Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Defines two functions to reshape a list of multi-dimensional data into a data matrix. Note, that all samples are assumed to be of equal size.
def asRowMatrix(X):
    if len(X) == 0:
        return np.array([])
    mat = np.empty((0, X[0].size), dtype = X[0].dtype)
    for row in X:
        mat = np.vstack((mat, np.asarray(row).reshape(1, -1)))
    return mat

def asColumnMatrix(X):
    if len(X) == 0:
        return np.array([])
    mat = np.empty((X[0].size, 0), dtype = X[0].dtype)
    for col in X:
        mat = np.hstack((mat, np.asarray(col).reshape(-1, 1)))
    return mat

# The eigenvectors we have calculated can contain negative values, but the image data is excepted as unsigned integer values in the range of 0 to 255. So we need a function to normalize the data first
def normalize(X, low, high, dtype=None):
    X = np.asarray(X)
    minX, maxX = np.min(X), np.max(X)
    # normalize to [0...1].
    X = X - float(minX)
    X = X / float((maxX - minX))
    # scale to [low...high].
    X = X * (high-low)
    X = X + low
    if dtype is None:
        return np.asarray(X)
    return np.asarray(X, dtype=dtype)

def project(W, X, mu=None):
    if mu is None:
        return np.dot(X, W)
    return np.dot(X - mu, W)

def reconstruct(W, Y, mu=None):
    if mu is None:
        return np.dot(Y, W.T)
    return np.dot(Y, W.T) + mu

# Principal Component Analysis
def pca(X, y, num_components = 0):
    [n, d] = X.shape
    if (num_components <= 0) or (num_components > n):
        num_components = n

    # Calculating the mean
    mu = X.mean(axis = 0)
    X = X - mu

    # linalg.eigh: Return the eigenvalues and eigenvectors of a Hermitian or symmetric matrix.
    if n > d:
        C = np.dot(X.T, X)
        [eigenvalues, eigenvectors] = np.linalg.eigh(C)
    else:
        C = np.dot(X, X.T)
        [eigenvalues, eigenvectors] = np.linalg.eigh(C)
        eigenvectors = np.dot(X.T, eigenvectors)
        for i in xrange(n):
            eigenvectors[:, i] = eigenvectors[:, i] / np.linalg.norm(eigenvectors[:, i])

    # or simply perform an economy size decomposition
    # eigenvectors, eigenvalues, variance = np.linalg.svd(X.T, full_matrices=False)
    # sort eigenvectors descending by their eigenvalue

    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # select only num_components
    eigenvalues = eigenvalues[0 : num_components].copy()
    eigenvectors = eigenvectors[:, 0 : num_components].copy()
    return [eigenvalues, eigenvectors, mu]

# Reads image from the path we give it
def read_images(path, sz = None):
    c = 0
    X, y = [], []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    im = Image.open(os.path.join(subject_path, filename))
                    im = im.convert("L")
                    # resize to given size (if given)
                    if (sz is not None):
                        im = im.resize(sz, Image.ANTIALIAS)
                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(c)
                except IOError:
                    print "I/O error({0}): {1}".format(errno, strerror)
                except:
                    print "Unexpected error:", sys.exc_info()[0]
                    raise
            c = c + 1
    return [X, y]

# Visual Display
def create_font(fontname = 'Tahoma', fontsize = 10):
    return {
        'fontname': fontname,
        'fontsize': fontsize
    }

def subplot(title, images, rows, cols, sptitle = "subplot", sptitles = [], colormap = cm.gray, ticks_visible = True, filename = None):
    fig = plt.figure()
    # main title
    fig.text(0.5, 0.95, title, horizontalalignment = 'center')
    for i in xrange(len(images)):
        ax0 = fig.add_subplot(rows, cols, (i + 1))
        plt.setp(ax0.get_xticklabels(), visible = False)
        plt.setp(ax0.get_yticklabels(), visible = False)
        if len(sptitles) == len(images):
            plt.title("%s #%s" % (sptitle, str(sptitles[i])), create_font('Tahoma', 10))
        else:
            plt.title("%s #%d" % (sptitle, (i + 1)), create_font('Tahoma', 10))
        plt.imshow(np.asarray(images[i]), cmap = colormap)
    if filename is None:
        plt.show()
    else:
        fig.savefig(filename)

# Distance Function
class AbstractDistance(object):
    def __init__(self, name):
        self._name = name
    def __call__(self, p, q):
        raise NotImplementedError("Every AbstractDistance must implement the __call__ method.")
    @property
    def name(self):
        return self._name
    def __repr__(self):
        return self._name

class EuclideanDistance(AbstractDistance):
    def __init__(self):
        AbstractDistance.__init__(self, "EuclideanDistance")
    def __call__(self, p, q):
        p = np.asarray(p).flatten()
        q = np.asarray(q).flatten()
        return np.sqrt(np.sum(np.power((p-q), 2)))

def visualizeEigenfaces():
    # Visualize the Data
    E = []
    for i in xrange(min(len(X), 16)):
        e = W[:,i].reshape(X[0].shape)
        E.append(normalize(e, 0, 255))

    # plot them and store the plot to "python_eigenfaces.pdf"
    subplot(title = "Eigenfaces AT&T Facedatabase", images = E, rows = 4, cols = 4, sptitle = "Eigenface", colormap = cm.jet, filename = "python_pca_eigenfaces.pdf")

def reconstructEigenfaces():
    # reconstruction steps
    steps = [i for i in xrange(10, min(len(X), 320), 20)]
    E = []
    for i in xrange(min(len(steps), 16)):
        numEvs = steps[i]
        P = project(W[:, 0: numEvs], X[0].reshape(1, -1), mu)
        R = reconstruct(W[:, 0: numEvs], P, mu)
        # reshape and append to plots
        R = R.reshape(X[0].shape)
        E.append(normalize(R, 0, 255))
    # plot them and store the plot to "python_reconstruction.pdf"
    subplot(title = "Reconstruction AT&T Facedatabase", images = E, rows = 4, cols = 4, sptitle = "Eigenvectors", sptitles = steps, colormap = cm.gray, filename = "python_pca_reconstruction.pdf")

if __name__ == '__main__':
    [X, y] = read_images("/Users/Abhi/Desktop/FacialRecognition/data/atnt")
    [D, W, mu] = pca(asRowMatrix(X), y)
