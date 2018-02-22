from sklearn import svm
from sklearn.model_selection import GridSearchCV
import os
import copy
import numpy as np
import pandas as pd
import logging
import time
from imgaug import augmenters as iaa
from sklearn.decomposition import PCA

#TODO error handling, check for correct data(types) (uint8 esp. for training)
#TODO testing
#TODO python3!!!!
#TODO GridSearch / optimize_hyperparameter() ?
#TODO module visualization ?
#TODO parallelize

class TextureClassifier(object):

    def __init__(self, num_histogram_bins=10, pca_dims=10, kernel="rbf", probability=True):
        """ Initialize. """
        self.num_histogram_bins = num_histogram_bins
        self.pca_dims = pca_dims
        self.with_probability = probability
        self.classifier = svm.SVC(kernel=kernel, probability=probability)
        self.feature_extractor = HistogramFeatureExtractor(self.num_histogram_bins)
        self.pca = None
        self.traindata = None
        self.logger = logging.getLogger(__package__)

    @classmethod
    def load(cls, filename):
        """ Load trained classifier from pickle file.
        Args:
            filename (str): Filename containing pickled classifier.
        """
        import pickle
        dictionary = pickle.load(open(filename))
        obj = cls(dictionary["histogram_bins"], dictionary["numFeatures"])  # old names (from old classifiers)...
        #obj = cls(dictionary["num_histogram_bins"], dictionary["pca_dims"])
        obj.pca = dictionary["pca"]
        if type(dictionary["svm"]) == GridSearchCV:
            obj.classifier = dictionary["svm"].best_estimator_   # for old classifiers
        else:
            obj.classifier = dictionary["svm"]
        obj.with_probability = obj.classifier.probability
        return obj

    @staticmethod
    def _init_augmentor():
        """ Init image augmentor. """
        st = lambda aug: iaa.Sometimes(0.5, aug)
        augmentor = iaa.Sequential([iaa.Crop(px=(0, 25)),  # crop images from each side by 0 to 16px (randomly chosen)
                                    iaa.Fliplr(0.5),       # horizontally flip 50% of the images
                                    iaa.Flipud(0.5),
                                    st(iaa.Add((-80,40),per_channel=False)),
                                    st(iaa.Multiply((0.7,1.5),per_channel=False)),
                                    st(iaa.AdditiveGaussianNoise(scale=(0.01*255,0.03*255))),
                                    st(iaa.Affine(scale={"x":(1.0,1.3),"y":(1.0,1.3)})),
                                    st(iaa.Affine(rotate=(90))),
                                    st(iaa.Affine(rotate=(90))),
                                    st(iaa.Affine(rotate=(90))),
                                    st(iaa.ContrastNormalization((0.7,1.5)))],
                                   random_order=True)
        return augmentor

    def _augment_patches(self, num_augmentations):
        """ Augment train patches. """
        if num_augmentations < 1:
            return
        start = time.time()
        self.logger.info("Start image augmentation...")
        self.augmentor = self._init_augmentor()
        augmented = []
        labels = []
        # if all patches have same shape:
        #patches, labels = self.traindata[["patches", "labels"]].values
        #augmented = len(self.augmentor.augment_images(patches.tolist()*num_augmentations))
        #labels = labels.tolist()*num_augmentations
        for patch, label in self.traindata[["patches","labels"]].values:
            augmented = augmented + self.augmentor.augment_images([patch]*num_augmentations)
            labels = labels + [label]*num_augmentations
        augmented_data = pd.DataFrame(data={"patches":augmented, "labels":labels, "augmented":[True]*len(labels)})
        self.traindata = self.traindata.append(augmented_data, ignore_index=True)
        self.logger.info("Image augmentation took {} seconds. Number of patches after augmentation: {}".format(time.time()-start,
                                                                                                               len(self.traindata)))

    def _fit_pca(self):
        """ Fit and apply PCA to train data. """
        start = time.time()
        self.logger.info("Start fitting PCA...")
        pca = PCA(n_components=self.pca_dims)
        tmp = self.traindata["features"].values
        features = np.array([t for t in tmp])
        pca.fit(features)
        self.traindata["features"] = pd.Series(data=pca.transform(features).tolist())
        self.pca = pca
        self.logger.info("Fitting PCA took {} seconds.".format(time.time()-start))

    def _apply_pca(self, features):
        """ Apply trained PCA to new features.
        Args:
            features (list): List of features with shape (n_samples, n_features).
        Returns:
            list: List of features with reduced dimensionality with shape (n_samples, n_components).
        """
        features = np.array([f for f in features])
        transformed = self.pca.transform(features)
        return transformed.tolist()

    def _fit_svm(self):
        """ Fit SVM to features generated from train data. """
        start = time.time()
        self.logger.info("Start fitting SVM...")
        #TODO inwiefern soll grid search durch benutzer steuerbar sein? -> methode (_?)optimize_hyperparameter()?
        param_grid = {"C": 10.**np.arange(-3, 8), "gamma": 10.**np.arange(-5, 4)}
        sv = GridSearchCV(self.classifier, param_grid, cv=3, refit=True, n_jobs=-1)  # with 3fold cross_validation
        sv.fit([f for f in self.traindata["features"].values], [l for l in self.traindata["labels"].values])
        #self.classifier.fit([f for f in self.traindata["features"].values], [l for l in self.traindata["labels"].values])
        print "Best parameters:",sv.best_params_
        print "Best score:",sv.best_score_
        self.classifier = sv.best_estimator_
        self.logger.info("Fitting SVM took {} seconds.".format(time.time()-start))

    def train(self, patches, labels, num_augmentations=5):
        """ Train classifier.
        Args:
            patches (list): List of train patches.
            labels (list) Labels according to given patch list.
        """
        assert len(patches) == len(labels), "Length of label list and length of patch list must be equal."
        # save original and augmented train data
        self.traindata = pd.DataFrame(data={"patches":patches, "labels":labels, "augmented":[False]*len(labels)})
        self._augment_patches(num_augmentations)

        # extract features from patches
        start = time.time()
        self.logger.info("Start feature extraction...")
        patches = self.traindata["patches"].values
        features = self.feature_extractor.extract_features(patches)
        self.traindata["features"] = pd.Series(data=features)
        self.logger.info("Feature extraction took {} seconds.".format(time.time()-start))

        # fit PCA and SVM
        self._fit_pca()
        self._fit_svm()

    def predict(self, patches, probability=False):
        """ Predict labels for given patches.
        Args:
            patches (list): List of 2D image patches.
            probability (bool): Predict class probabilities if SVM was trained with probability=True.
        Returns:
            list: Predicted labels for given patches with length n_patches.
                  If probability: Predicted class probabilities for given patches with shape (n_patches, n_classes).
        """
        # extract features from patches
        features = self.feature_extractor.extract_features(patches)
        features = self._apply_pca(features)
        features = np.array([f for f in features])
        if probability and self.with_probability:
            labels = self.classifier.predict_proba(features)
        else:
            labels = self.classifier.predict(features)
        return labels.tolist(), features.tolist()

    def predict_label(self, patches, label, min_probability=0.5):
        """ Predict given label for given patches if label probability > min_probability.
        Args:
            patches (list): List of 2D image patches.
            label (int): Trained class label.
            min_probability (float): Minimal probability for given class label to be predicted as True.
        Returns:
            list: Label probability per patch if label probability > given min_probability, else 0.

        """
        assert label in self.classifier.classes_, "Given class label {} has not been trained. Trained class labels are: {}".format(label, self.classifier.classes_)
        if len(patches) == 0:
            return []
        proba, features = self.predict(patches, probability=self.with_probability)
        if type(proba[0]) == int:  # predicted labels instead of probabilities
            return [1 if p == label else 0 for p in proba], features
        else:
            label_idx = np.where(self.classifier.classes_ == label)[0][0]
            return [p[label_idx] if p[label_idx] >= min_probability else 0 for p in proba], features

    @staticmethod
    def get_grid_patches(image, gridsize, patchsize, gridpoints=None):
        """ Get image patches of size patchsize on gridpoints with given gridsize.
        Args:
            image (ndarray): Image.
            gridsize (int): Distance between gridpoints.
            patchsize (int): Size of cropped patches.
            gridpoints (list, optional): Predefined gridpoint coordinates in form [[x1,y1],[x2,y2],...].
        Returns:
            list: Cropped patches.
            list: X and Y coordinates of cropped patches.
        """
        patches = []
        if gridpoints is None:
            gridpoints = [[x,y] for x in range(patchsize/2, image.shape[1]-patchsize/2, gridsize)
                                for y in range(patchsize/2, image.shape[0]-patchsize/2, gridsize)]
        for x,y in gridpoints:
            patches.append(image[y-patchsize/2:y+patchsize/2, x-patchsize/2:x+patchsize/2])
        return patches, gridpoints

    #TODO MeanShift
    def detect(self, image, label, gridsize=40, min_feature_size=24, max_feature_size=80, min_probability=0.5):
        """ Detect features in given image.
        Args:
            image (ndarray): Image (should have same spacing as train data).
            label (int): Label of class to be detected.
            gridsize (int, optional): Size of grid used to sample image patches.
            min_feature_size (int, optional): Minimal expected size of features to detect.
            max_feature_size (int, optional) Maximal expected size of features to detect.
            min_probability (float): Minimal probability for given class label to be predicted as True.
        Returns:
            pandas.DataFrame: Detected landmarks with (x, y, size, feature_vector, probability)
        """
        assert label in self.classifier.classes_, "Given class label {} has not been trained. Trained class labels are: {}".format(label, self.classifier.classes_)
        assert image.dtype == np.uint8, "Given image must be of type uint8, but is type: {}".format(image.dtype)
        num_steps = 6
        min_detections = 2
        gridpoints = None  # in first call of get_grid_patches, gridsize is None to define new grid
        for patchsize in np.linspace(max_feature_size, min_feature_size, num_steps, dtype=np.uint8):
            print patchsize
            patches, gridpoints = self.get_grid_patches(image, gridsize, patchsize=patchsize, gridpoints=gridpoints)
            detected, features = self.predict_label(patches, label, min_probability=min_probability)
            detected = np.array(detected)
            detected[detected > 0] = 1
            if patchsize == max_feature_size:
                detected_sum = detected.astype(np.uint8)  # Initialize detected_sum
            else:
                detected_sum += detected.astype(np.uint8)  # sum up scores in detected
        XY = np.array(gridpoints)[detected_sum >= min_detections]  # at least 2 detections in num_steps scales
        # TODO calculate proper probas, features and sizes!!
        probabilities = np.array(detected_sum)[detected_sum >= min_detections].tolist()
        features = np.array(features)[detected_sum >= min_detections].tolist()
        return pd.DataFrame(data={"x":XY[:,0].tolist(), "y":XY[:,1].tolist(), "size":[patchsize]*len(XY),
                                  "feature_vector":features, "probability":probabilities})

    def save(self, filename, overwrite=False):
        """ Save trained classifier.
        Args:
            filename (str): Filename for saving classifier (should has suffix '.p'??).
            overwrite (bool, optional): If True: Overwrite file if already existing.
        """
        import pickle
        if not overwrite and os.path.exists(filename):
            raise IOError("Given filename already exists. Set overwrite=True to overwrite existing file.")
        #dictionary = {"num_histogram_bins":self.num_histogram_bins, "pca_dims":self.pca_dims, "svm": self.classifier, "pca": self.pca}
        # old format:
        dictionary = {"histogram_bins":self.num_histogram_bins, "numFeatures":self.pca_dims, "svm": self.classifier, "pca": self.pca,
                      "labels":{"vessels":1,"bg":0}, "train_spacing":0.5, "defaultFeatureSizes":{"vessels":[150,400]}}
        pickle.dump(dictionary, open(filename,"w"))
        self.logger.info("Saved classifier to {}".format(filename))


class HistogramFeatureExtractor(object):

        def __init__(self, num_histogram_bins, maxSize=500):
            """ Initialize. """
            self.num_histogram_bins = num_histogram_bins
            self.radius_img = self._create_radius_image(maxSize)

        @staticmethod
        def _create_radius_image(size):
            size = int(size)
            radius_img = np.zeros((size,size))
            center = [int(size/2.), int(size/2.)]
            for y in range(size):
                for x in range(size):
                    radius_img[y,x] = np.sqrt(((np.array([center[0],center[1]])-np.array([y,x]))**2).sum())
            return radius_img

        def _get_radius_image(self):
            """ Get copy of self.radius_img. """
            return copy.deepcopy(self.radius_img)

        def _get_resized_radius_image(self, shape):
            """ Get radius image with given shape. """
            difference = (np.array(self.radius_img.shape) - np.array(shape))/2.
            if (difference < 0).any():  # enlarge self.radius_img
                self.radius_img = self._create_radius_image(max(shape)*1.5)
            radius_img = self._get_radius_image()
            if difference[0] > 0:
                radius_img = radius_img[int(difference[0]):-int(difference[0]+0.5),:]
            if difference[1] > 0:
                radius_img = radius_img[:,int(difference[1]):-int(difference[1]+0.5)]
            return radius_img

        def extract_features(self, patches):
            """ Extract 2D histogram features from given patches.
            Args:
                patches (list): Image patches.
            Returns:
                list: List of patch features of length self.num_histogram_bins**2.
            """
            features = []
            for patch in patches:
                radius_img = self._get_resized_radius_image(patch.shape)
                assert patch.shape == radius_img.shape, "ERROR: patch.shape ({}) != radiusImg.shape ({})".format(patch.shape, radius_img.shape)
                # calculate and normalize 2D histogram
                hist, _, _ = np.histogram2d(radius_img.flatten(), patch.flatten(), bins=[self.num_histogram_bins,np.linspace(0,256,num=self.num_histogram_bins+1)])
                features.append((hist / float(patch.shape[0]*patch.shape[1])).flatten())
            return features
