import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances

A = np.mat("10, 60, 10, 90; 20, 50, 40, 70; 30, 50, 30, 40; 20, 50, 20, 60; 10, 60, 30, 10")
print("Matrix is:\n", A)

#   a)
norm = np.apply_along_axis(np.linalg.norm, 1, A)
print("Norm is:\n", norm)

#---------------------------------------------------------------------------------------------

#   b)
similarities = cosine_similarity(A)
print("Similarities:\n", similarities)

#---------------------------------------------------------------------------------------------

#   c)
euclideanDistance = euclidean_distances(A, A)
print("Euclidean Distance:\n", euclideanDistance)
