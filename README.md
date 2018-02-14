# demics
## Usage
Train classifier:
```python
classifier = HistogramSVM(num_bins=10, pca_dims=10)
patches = [p1,p2,...,n1,n2,...] # patches of different classes
labels = [1,1,...,0,0,...] # label per patch
classifier.train(patches, labels)
```

Predict image patches:
```python
candidates = [c1, c2, ...] # patches
labels = classifier.predict(candidates)
```

Detect features in image:
```python
features = classifier.detect(img, gridsize=40, min_feature_size=120, max_feature_size=400)
```
