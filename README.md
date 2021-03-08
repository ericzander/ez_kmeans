# ez_kmeans

## Description

Custom k-means clustering module with demos of applications in data analysis and image processing.

## How to Use

Models are instantiated like this:
> km_model = kmeans(k=6)

This would create a model with 6 centroids.
The model can then be fit to data like this:
> km_model.fit(training_data)

Fitting also returns a numpy array of cluster labels for the fitted data.
These labels and the centroids can be retrieved:
> centroids = km_model.centroids

> labels = km_model.labels

Finally, novel similar data can be clustered the following way:
> predictions = km_model.predict(prediction_data)

## Contents

*kmeans.py*
> Custom k-means module.

*MusicExample/MusicClustering.ipynb*
> Basic demo of clustering with continuous data extracted from 30 second song snippets.

*FlowerImageExample/FlowerClustering.ipynb*
> Demo of basic image segmentation and vector quantization for compression with flower images.

## Potential Additions
* Demo k value selection methods (ex. elbow method)
* Include better data analysis application example
* Experiment with other centroid initialization methods
