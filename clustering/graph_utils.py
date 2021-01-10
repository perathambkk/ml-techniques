"""
Author: Peratham Wiriyathammabhum


"""
import numpy as np 
import pandas as pd 
from sklearn.neighbors import NearestNeighbors

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

def sparse_affinity_graph(X):
	'''
	TODO: This function returns a numpy sparse matrix.
	'''
	ni, nd = X.shape
	A = np.zeros((ni, ni))
	for i in range(ni):
		for j in range(i+1, ni):
			dist = ((X[i] - X[j])**2).sum() # compute L2 distance
			A[i][j] = dist
			A[j][i] = dist # by symmetry
	return A

def laplacian_graph(X, mode='affinity', knn=3, eta=0.01, sigma=2.5):
	'''
	The unnormalized graph Laplacian, L = D − W.
	'''
	if mode == 'affinity':
		W = affinity_graph(X)
		W[abs(W) > eta] = 0
	elif mode == 'nearestneighbor':
		W = knn_graph(X, knn=knn)
	elif mode == 'gaussian':
		W = affinity_graph(X)
		bandwidth = 2.0*(sigma**2)
		W = np.exp(W) / bandwidth
	else:
		pass
	D = np.diag(W.sum(axis=1))
	L = D - W
	return L
	