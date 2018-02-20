# demics
## Usage
Train classifier:
```python
classifier = HistogramSVM(num_bins=10, pca_dims=10)
patches = [p1,p2,...,n1,n2,...] # patches of different classes
labels = [1,1,...,0,0,...] # label per patch
classifier.train(patches, labels, num_augmentations=2)
classifier.save("demics_classifier.p")
```

Predict image patches:
```python
classifier = TextureClassifier.load("demics_classifier.p")
candidates = [c1, c2, ...] # patches
labels = classifier.predict(candidates)
label_probabilities = classifier.predict_label(candidates, label=1, min_probability=0.5)
```

Detect features in image:
```python
detections = classifier.detect(img, label=1, gridsize=40, min_feature_size=120, max_feature_size=400, min_probability=0.7)
```
