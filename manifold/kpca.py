"""
Author: Peratham Wiriyathammabhum


"""
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy import linalg as SLA
from numpy import linalg as LA
from sklearn.utils.extmath import row_norms, safe_sparse_dot

def affinity_graph(X):
	'''
	This function returns a numpy array.
	'''
	ni, nd = X.shape
	A = np.zeros((ni, ni))
	for i in range(ni):
		for j in range(i+1, ni):
			dist = ((X[i] - X[j])**2).sum() # compute L2 distance
			A[i][j] = dist
			A[j][i] = dist # by symmetry
	return A

def knn_graph(X, knn=4):
	'''
	This function returns a numpy array.
	'''
	ni, nd = X.shape
	nbrs = NearestNeighbors(n_neighbors=(knn+1), algorithm='ball_tree').fit(X)
	distances, indices = nbrs.kneighbors(X)
	A = np.zeros((ni, ni))
	for dist, ind in zip(distances, indices):
		i0 = ind[0]
		for i in range(1,knn+1):
			d = dist[i]
			A[i0, i] = d
			A[i, i0] = d # by symmetry
	return A

def _apply_kernel(Xin, kernel="poly", params={}):
	if kernel not in ["poly","rbf","tanh"]:
		raise NotImplementedError("Not implemented Error.")
	
	X = Xin.copy()

	if kernel == "poly":
		if "coef0" not in params.keys() or "degree" not in params.keys() or "gamma" not in params.keys():
			raise KeyError("KeyError.")
		# K(X, Y) = (gamma <X, Y> + coef0) ^ degree See: https://github.com/scikit-learn/scikit-learn/blob/fe2edb3cd/sklearn/metrics/pairwise.py#L1749
		KXres = safe_sparse_dot(X.T, X, dense_output=True) # dot_product implemented in sklearn
		KXres *= params["gamma"]
		KXres += params["coef0"]
		KXres **= params["degree"]
		pass
	elif kernel =="rbf":
		# K(x, y) = exp(-gamma ||x-y||^2)
		pass
	elif kernel == "tanh":
		if "coef0" not in params.keys() or "gamma" not in params.keys():
			raise KeyError("KeyError.")
		# K(X, Y) = tanh(gamma <X, Y> + coef0) See: https://github.com/scikit-learn/scikit-learn/blob/fe2edb3cd/sklearn/metrics/pairwise.py#L1749
		KXres = safe_sparse_dot(X.T, X, dense_output=True) # dot_product implemented in sklearn
		KXres *= params["gamma"]
		KXres += params["coef0"]
		KXres = np.tanh(KXres)

	# Normalize kernel matrix K
	ell = X.shape[0]
	column_sums = np.sum(KXres, axis=1) / ell                       # column sums
	total_sum   = sum(column_sums) / ell     # total sum
	J = np.ones([ell, 1]) * column_sums                  # column sums (in matrix)
	KXres = KXres - J - J.T
	KXres = KXres + total_sum

	return KXres

def kpca(Xin):
	"""
	Perform Kernel PCA on an input row matrix X.
	
	"""
	X = Xin.copy() # for safety
	ni, nd = X.shape

	# centering
	X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

	# Kernel
	kernel_type = "poly"
	kernel_params = {}

	n_feas = X.shape[1]
	if kernel_type == "poly":
		kernel_params = {"degree":3, "coef0":1.0, "gamma":1.0/n_feas}
	KX = _apply_kernel(X, kernel=kernel_type, params=kernel_params)

	w, v = LA.eigh(KX)
	ind = np.argsort(-w, axis=0) # sorting descending
	w = w[ind]
	v = v[:, ind]
	
	proj_v = v[:,0:2] # projection matrix
	X_r = np.matmul(KX, proj_v)
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