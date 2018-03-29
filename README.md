# demics

## Dependencies
* numpy
* scikit-learn
* pandas
* imgaug

## Installation

### Installation on Linux / Mac
Clone the repository from github and install with pip:

```bash
git clone https://github.com/FZJ-INM1-BDA/demics.git
cd demics
pip install -r requirements.txt
pip install .
```

## Development
Clone demics from github and install with option -e, so that no reinstallation is needed for every update.
```bash
git clone https://github.com/FZJ-INM1-BDA/demics.git
cd demics
# git checkout develop
pip install -r requirements.txt
pip install -e .
```

## Usage
Train classifier:
```python
classifier = TextureClassifier(num_histogram_bins=10, pca_dims=10)
patches = [p1,p2,...,n1,n2,...]  # patches of different classes
labels = [1,1,...,0,0,...]  # label per patch
# optional: define parameters for Grid Search
param_grid = {"C": 10.**np.arange(-3, 8), "gamma": 10.**np.arange(-5, 4)}
classifier.grid_search(param_grid, n_jobs=-1, cv=3, refit=True)
# start training (with Grid Search if defined)
classifier.train(patches, labels, num_augmentations=2)
classifier.save("demics_classifier.p")
```

Predict image patches:
```python
classifier = TextureClassifier.load("demics_classifier.p")
candidates = [c1, c2, ...]  # patches
labels, scores = classifier.predict(candidates)
```

Detect features in image:
```python
detections = classifier.detect(img, label=1, gridsize=40, min_feature_size=120, max_feature_size=400, min_probability=0.7)
```
