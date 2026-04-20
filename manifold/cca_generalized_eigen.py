"""
Author: Peratham Wiriyathammabhum


"""
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
# import scipy.sparse.linalg as linalg
from numpy import linalg as LA
from scipy import linalg as SLA

# Import the libraries
# from sklearn.cross_decomposition import CCA

def cca(Xin, Yin):
	"""
	Perform Canonical Correlation Analysis (CCA) on an input row matrix X ,Y.
	
	See: Hardoon, D. R.; Szedmak, S.; Shawe-Taylor, J. (2004). "Canonical Correlation Analysis: An Overview with Application to Learning Methods". Neural Computation. 16 (12): 2639–2664
		https://en.wikipedia.org/wiki/Canonical_correlation
		https://github.com/scikit-learn/scikit-learn/blob/a95203b249c1cf392f86d001ad999e29b2392739/sklearn/cross_decomposition/pls_.py
		https://github.com/scikit-learn/scikit-learn/blob/a95203b/sklearn/cross_decomposition/cca_.py#L6

	"""
	X = Xin.copy() # for safety
	Y = Yin.copy() # for safety
	ni, nd = X.shape

	# centering
	X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
	Y = (Y - np.mean(Y, axis=0)) / np.std(Y, axis=0)

	# cross correlation
	Cxy = np.dot(X.T, Y)
	Cxx = np.dot(X.T, X)
	Cyx = np.dot(Y.T, X)
	Cyy = np.dot(Y.T, Y)

	# eigen decomposition
	A = np.dot(LA.pinv(Cyy, hermitian=True), Cyx)
	A = np.dot(Cxy, A)
	B = Cxx
	sx, ux = SLA.eig(A, b=B)

	A = np.dot(LA.pinv(Cxx, hermitian=True), Cxy)
	A = np.dot(Cyx, A)
	B = Cyy
	sy, uy = SLA.eig(A, b=B)

	ind = np.argsort(sx, axis=0) # sorting descending
	ux = ux[:, ind]
	ind = np.argsort(sy, axis=0) # sorting descending
	uy = uy[:, ind]
	
	# create coordinate matrix
	proj_ux = ux[:,0:2] # projection matrix
	proj_uy = uy[:,0:2] # projection matrix
	X_r = np.matmul(X, proj_ux)
	Y_r = np.matmul(Y, proj_uy)

	return X_r, Y_r

def main():
	# load data
	# from sklearn import datasets
	# Set the random seed for reproducibility
	np.random.seed(0)

	# Generate X and Y with 10 dimensions each
	X = np.random.randn(100, 10)
	Y = X + np.random.randn(100, 10)	

	# timer
	from time import time
	t0 = time()
	# dimred
	X_r, Y_r = cca(X, Y)
	t1 = time()
	print('[Info] Canonical Component Analysis done in {:.2g} sec.'.format(t1 - t0))

	# plot
	fig = plt.figure()

	ax = fig.add_subplot(221)
	ax.scatter(X[:, 0], X[:, 1])
	ax.set_title("Original data X")
	plt.xticks([]), plt.yticks([])

	ax = fig.add_subplot(222)
	ax.scatter(Y[:, 0], Y[:, 1])
	ax.set_title("Original data Y")
	plt.xticks([]), plt.yticks([])

	ax = fig.add_subplot(223)
	ax.scatter(X_r[:, 0], X_r[:, 1])
	ax.set_title("Projected data X")
	plt.xticks([]), plt.yticks([])

	ax = fig.add_subplot(224)
	ax.scatter(Y_r[:, 0], Y_r[:, 1])
	ax.set_title("Projected data Y")
	plt.xticks([]), plt.yticks([])

	# plt.axis('tight')
	# plt.title('Projected data')
	plt.show()
	input("Press Enter to continue...")

	# correlation scores
	score = X_r.T @ Y_r
	corr_score = np.sum(np.abs(score[:]))/len(score[:].flatten())
	input("Correlation score: {}".format(corr_score))
	
	return

if __name__ == '__main__':
	main()