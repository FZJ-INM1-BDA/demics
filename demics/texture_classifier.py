from sklearn import svm
import os

class TextureClassifier(object):

    def __init__(self, num_histogram_bins=10, pca_dims=10, kernel="rbf", probability=True):
        """ Initialize Object. """
        self.num_bins = num_histogram_bins
        self.pca_dims = pca_dims
        self.classifier = svm.SVC(kernel=kernel, probability=probability)
        self.pca = None

    @classmethod
    def load(cls, filename):
        """ Load trained classifier from pickle file.
        Args:
            filename (str): Filename containing pickled classifier.
        """
        import pickle
        dictionary = pickle.load(open(filename))
        cls(dictionary["num_histogram_bins"], dictionary["pca_dims"], dictionary["kernel"], dictionary["probability"])

    def _augment_patches(self):
        pass

    def _extract_features(self):
        pass

    def _fit_pca(self):
        pass

    def _apply_pca(self):
        pass

    def _fit_svm(self):
        pass

    def train(self):
        """ Train classifier. """
        pass

    def predict(self, patches):
        """ Predict labels for given patches.
        Args:
            patches (list): List of 2D image patches.
        Returns:
            list: Predicted labels of given patches.
        """
        pass

    def detect(self, image, gridsize=40, min_feature_size=120, max_feature_size=400):
        """ Detect features in given image.
        Args:
            image (ndarray): Image.
            gridsize (int, optional): Size of grid used to sample image patches.
            min_feature_size (int, optional): Minimal expected size of features to detect.
            max_feature_size (int, optional) Maximal expected size of features to detect.
        """
        pass

    def save(self, filename, overwrite=False):
        """ Save trained classifier.
        Args:
            filename (str): Filename for saving classifier (should has suffix '.p'??).
            overwrite (bool, optional): If True: Overwrite file if already existing.
        """
        import pickle
        if not overwrite and os.path.exists(filename):
            raise IOError("Given filename already exists. Set overwrite=True to overwrite existing file.")
        dictionary = {"num_histogram_bins":self.num_histogram_bins, "pca_dims":self.pca_dims, "svm": self.classifier, "pca": self.pca}
        pickle.dump(dictionary, open(filename,"w"))
