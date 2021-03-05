"""This module contains a simple custom k-means clustering class.

When fit to a 1-D or 2-D array of data, calculates k centroids and returns
cluster labels for each point. The model retains the centroids, and can
then predict the cluster labels for novel data.

    Example:

        km = kmeans(k=4)
        clusters = km.fit(data)
        predictions = km.predict(new_data)

Read more about k-means: https://en.wikipedia.org/wiki/K-means_clustering

Author: Eric Zander
Email:  ericzander.ml@gmail.com
"""

import numpy as np


class kmeans:
    """k-means clustering model fitting and predicting.

    Attributes:
        k: The number of clusters.
        centroids: numpy array of centroids.
        labels: numpy array of fitted data cluster labels.
    """

    def __init__(self, k):
        """
        Initialize k-means clustering model.

        Parameters:
            k: The number of clusters
        """
        self.k = k
        self.centroids = None
        self.labels = None

    def fit(self, X, max_iter=300, seed=None):
        """Fits the model to given data and returns cluster labels.

        Given data can be 1-D or 2-D numpy arrays, a pandas Series, a
        pandas DataFrame, or other similar format that can be converted
        into a numpy array.

        Parameters:
            X: The data to fit the model to.
            max_iter: The max amount of iterations for centroid updating.
            seed: Seed for random initialization of clusters.

        Returns:
            numpy ndarray with cluster labels for each point.
        """
        # Seed the random number generator if seed supplied
        if seed is not None:
            np.random.seed(seed)

        # Convert training data to numpy array and find max values for RNG
        X = np.array(X)
        max_val = X.max()

        # Initialize centroids and prepare to count iterations
        centroids = np.array([max_val * np.random.random_sample(X.shape[1])
                              for i in range(self.k)])
        p_centroids = np.full(centroids.shape, -1)
        iterations = 0

        # Loop until centroids aren't updated between iterations or max iter met
        while not self.__should_end(p_centroids, centroids, iterations, max_iter):
            # Update previous centroids and iterations
            p_centroids = np.copy(centroids)
            iterations = iterations + 1

            # Calculate distance squared from centroids (xn - kn)^2
            # Add dimension to centroids to use numpy broadcasting
            # https://numpy.org/doc/stable/user/basics.broadcasting.html
            dist = ((X - centroids[:, np.newaxis])**2).sum(axis=2)

            # Label data with index of nearest centroid
            # Label is the index of smallest distance in each row
            labels = np.argmin(dist, axis=0)

            # Update all centroids
            self.__update_centroids(X, centroids, labels, max_val)

        # Save centroids and labels
        self.centroids = np.copy(centroids)
        self.labels = np.copy(labels)

        return self.labels

    def predict(self, X):
        """Predicts cluster for each row using centroids of fitted model.

        Parameters:
            X: Array of data of similar dimensions to that used to fit the model.

        Returns:
            numpy of cluster labels for new data using fitted model
        """
        if self.centroids is None:
            raise ModelNotFitted("Model needs to be fitted to data")

        dist = ((X - self.centroids[:, np.newaxis])**2).sum(axis=2)
        labels = np.argmin(dist, axis=0)

        return labels

    def __should_end(self, p_centroids, centroids, iterations, max_iter):
        """Determines if centroid adjustment should end.

        Returns True if centroids not updating or max iterations reached.
        Otherwise, returns False.

        Parameters:
            p_centroids: Previous centroids
            centroids: Current centroids
            iterations: Number of iterations so far
            max_iter: Maximum number of iterations

        Returns:
            Boolean value indicating if kmeans fitting should end.
        """
        # Reshape for comparison
        p_centroids = p_centroids.reshape(-1)
        centroids = centroids.reshape(-1)

        # Return true if centroids didn't change or max iterations met
        if np.allclose(p_centroids, centroids, atol=0.0001):
            return True
        elif iterations >= max_iter:
            return True

        # Return false if conditions not met
        return False

    def __update_centroids(self, X, centroids, labels, max_val):
        """Updates centroids every iteration."""
        # If any centroid has no points, randomly reassign
        empty_centroids = [i for i in range(
            self.k) if i not in np.unique(labels)]
        for i in empty_centroids:
            centroids[i] = max_val * np.random.random_sample(X.shape[1])

        # Recalculate new positions of all other centroids
        # Positions are the mean of all points belonging to each centroid
        for i in range(len(centroids)):
            if i not in empty_centroids:
                centroids[i] = np.array(
                    X[np.where(labels == i)].mean(axis=0))


class ModelNotFitted(Exception):
    """
    Exception raised when attempting to call a method before a model is fitted.
    """
