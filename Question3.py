import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import linalg as LA

#   a)
A = np.mat("10, 60, 90; 20, 50, 70; 30, 50, 40; 20, 50, 60; 10, 60, 10")
print("Matrix is:\n", A)

#---------------------------------------------------------------------------------------------

#   b)
figure = plt.figure()
ax = figure.add_subplot(111, projection='3d')
ax.scatter(A[:, 0], A[:, 1], A[:, 2])
plt.show()

#---------------------------------------------------------------------------------------------

#   c)
mean = np.mean(A, axis=0)
print("Mean = ", mean)

#---------------------------------------------------------------------------------------------

#   d)
Z = A - mean
print("Centered data matrix Z:\n", Z)

#---------------------------------------------------------------------------------------------

#   e)
AT = np.transpose(A)
cov = np.cov(AT)
print("Covariance Matrix:\n", cov)

#---------------------------------------------------------------------------------------------

#   f)
eigenvalues, eigenvectors = LA.eig(cov)
print("Eigenvalues:\n", eigenvalues)
print("Eigenvectors:\n", eigenvectors)
diagonalMatrix = np.diag(eigenvalues)
print("Diagonal Matrix:\n", diagonalMatrix)

#---------------------------------------------------------------------------------------------

#   g)
UT = np.transpose(eigenvectors)
result = np.dot(UT, diagonalMatrix)
finalresult = np.dot(result, eigenvectors)
print("Final Result:\n", finalresult)

#---------------------------------------------------------------------------------------------

#   h)
variance = np.var(eigenvectors[:, 2], axis=0, )
print("Variance = ", variance)
# No, One eigenvector is not enough.

#---------------------------------------------------------------------------------------------

#   i)
P = np.column_stack((eigenvectors[:, 0], eigenvectors[:, 2]))
print(P)

#---------------------------------------------------------------------------------------------

#   j)
transformed = P.T.dot(AT)
print("Transformed:\n", transformed)

#---------------------------------------------------------------------------------------------

#   k)
transT = np.transpose(transformed)
print("transT:\n", transT)
plt.scatter([transT[:, 0]], [transT[:, 1]])
plt.show()