"""
Author: Peratham Wiriyathammabhum


"""
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
# import scipy.sparse.linalg as linalg
from numpy import linalg as LA

def pca(X):
	"""
	Perform PCA on an input row matrix X.
	
	"""
	ni, nd = X.shape

	# TODO: PCA for small samples where nd >> ni
	C = np.cov(X.T)

	w, v = LA.eig(C)
	ind = np.argsort(-w, axis=0) # sorting descending
	w = w[ind]
	v = v[:, ind]
	
	proj_v = v[:,0:2] # projection matrix
	X_r = np.matmul(X, proj_v)
	return X_r

def main():
	# load data
	from sklearn import datasets
	X, color = datasets.make_swiss_roll(n_samples=2500)

	# dimred
	X_r = pca(X)

	# plot
	fig = plt.figure()

	ax = fig.add_subplot(211, projection='3d')
	ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)

	ax.set_title("Original data")
	ax = fig.add_subplot(212)
	ax.scatter(X_r[:, 0], X_r[:, 1], c=color, cmap=plt.cm.Spectral)
	plt.axis('tight')
	plt.xticks([]), plt.yticks([])
	plt.title('Projected data')
	plt.show()
	input("Press Enter to continue...")
	
	return

if __name__ == '__main__':
	main()
