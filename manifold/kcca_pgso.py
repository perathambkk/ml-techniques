"""
Author: Peratham Wiriyathammabhum


"""
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

from scipy import linalg as SLA
from numpy import linalg as LA
from sklearn.utils.extmath import row_norms, safe_sparse_dot

def partial_gram_schmidth_orthogonolization(Kin, nu=0.00001):
	'''
	Partial Gram-Schmidth Orthogonolization (PGSO) described in Hardoon, D. R.;
	Szedmak, S.; Shawe-Taylor, J. (2004). "Canonical Correlation Analysis:
	An Overview with Application to Learning Methods". Neural Computation.
	16 (12): 2639–2664
	

	Inputs: K: an NxN matrix.
			nu: a precision parameter.
	'''
	K = Kin.copy()

	# initialization
	j = 0
	m = K.shape[0]
	size_vec = np.zeros(m)
	index_vec = np.zeros(m)
	feat = np.zeros([m, m])
	norm2 = np.diag(K)

	# algorithm
	while np.sum(norm2) > nu and j != N:
		ij = np.argmax(norm2)
		index_vec[j] = ij
		size_vec[j] = np.sqrt(norm2[ij])
		for i in range(m): # not vectorized
			sum_ff = 0.0
			for t in range(j):
				sum_ff += feat[i, t] * feat[ij, t]
			feat[i, j] = (K[i, ij] - sum_ff)/size_vec[j] # ?
			norm2[i] = norm2[i] - feat[i, j] ** 2
		j += 1

	return feat

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

def _sq_inverse(Xin):
	X = Xin.copy()
	X = 0.5 * (X + X.conj().T) # or X = np.maximum(X, X.T)

	L, V = np.linalg.eigh(X)
	
	L = np.diag(1.0/np.sqrt(L))
	
	Xres = V.conj() @ L @ V.T.conj()
	return Xres

def kcca_reg_chol(Xin, Yin, reg_kappa=0.01):
	"""
	Perform Regularized Kernel Canonical Correlation Analysis (RKCCA) on an input row matrix X ,Y.

	with a proper whitening and precise numerical routines as asked ChatGPT and Gemini -.-
	
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

	# Kernel
	kernel_type = "poly"
	kernel_params = {}

	n_feas = X.shape[1]
	if kernel_type == "poly":
		kernel_params = {"degree":3, "coef0":1.0, "gamma":1.0/n_feas}
	KX = _apply_kernel(X, kernel=kernel_type, params=kernel_params)
	if kernel_type == "poly":
		kernel_params = {"degree":3, "coef0":1.0, "gamma":1.0/n_feas}
	n_feas = Y.shape[1]
	KY = _apply_kernel(Y, kernel=kernel_type, params=kernel_params)

	# cross correlation
	_, S ,_ = partial_gram_schmidth_orthogonolization(KX)
	_, Rx ,_ = partial_gram_schmidth_orthogonolization(KX)
	_, Ry ,_ = partial_gram_schmidth_orthogonolization(KY)

	Zxx = Rx.T @ Rx 
	Zyy = Ry.T @ Ry
	Zxy = Rx.T @ Ry
	Zyx = Ry.T @ Rx

	# B = LA.pinv(shrink_KX) @ KY @ LA.pinv(shrink_KY) @ KX
	B = LA.pinv(S) @ Zxy @ LA.pinv(Zyy) @ Zyx @ LA.pinv(S.T) # scipy.linalg.solve_triangular(L, np.eye(L.shape[0]), lower=True)?
	su, u = LA.eigh(B)
	ind = np.argsort(-su, axis=0) # sorting descending
	u = u[:, ind]

	# B = LA.pinv(shrink_KY) @ KX @ LA.pinv(shrink_KX) @ KY
	_, S ,_ = partial_gram_schmidth_orthogonolization(KY)
	B = LA.pinv(S) @ Zyx @ LA.pinv(Zxx) @ Zxy @ LA.pinv(S.T)
	sv, v = LA.eigh(B)
	ind = np.argsort(-sv, axis=0) # sorting descending
	v = v[:, ind]

	
	# create coordinate matrix
	proj_v = v[:,0:2] # projection matrix
	proj_u = u[:,0:2] # projection matrix
	X_r = np.matmul(KX, proj_u)
	Y_r = np.matmul(KY, proj_v)

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
	X_r, Y_r = kcca_reg_chol(X, Y)
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