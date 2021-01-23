"""
Author: Peratham Wiriyathammabhum


"""
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
# import scipy.sparse.linalg as linalg
from numpy import linalg as LA

def fast_ica(Xin, n_components=3):
	"""
	Perform Fast Independent Component Analysis (FastICA) on an input row matrix X.
	X = AS, where A is a mixing matrix and S are signal sources.
	
	See: IJCNN'99 Independent Component Analysis: A Tutorial by Aapo Hyvrinen and Erkki Oja
		http://www.cs.jhu.edu/~ayuille/courses/Stat161-261-Spring14/HyvO00-icatut.pdf
	"""
	X = Xin.copy() # for safety
	ni, nd = X.shape
	
	"""
	Note: This X is a row matrix which is different to the original formulas which assume a column matrix.
	"""
	# centering
	X = (X - np.mean(X, axis=0)) 

	# whitening makes mixing matrix orthogonal
	D, E = LA.eig(np.cov(X.T))
	ind = np.argsort(-D, axis=0) # sorting descending
	D = D[ind]
	E = E[:, ind]
	D = np.power(D, -0.5)
	D = np.diag(D)
	X = E.dot(D).dot(E.T).dot(X.T)
	X = X.T

	# f, g, g' https://en.wikipedia.org/wiki/FastICA
	f = lambda x: np.log(np.cosh(x))
	g = lambda x: np.tanh(x)
	g_prime = lambda x: 1. - np.power(np.tanh(x), 2)

	W = []
	ones_m = np.ones((ni, 1))
	for i in range(n_components):
		print('[Info] Computing component {}...'.format(i))
		w_p = weight_init(nd, init_mode='random')
		w_p = w_p / np.linalg.norm(w_p, ord=2)

		TERMINATE = False
		while not TERMINATE:
			w_old = w_p.copy()
			first_term = (X.T.dot(g(X.dot(w_p))) / ni)
			second_term = ( g_prime(X.dot(w_p)).T.dot(ones_m) * (w_p) / ni)
			w_p = first_term - second_term
			if (len(W) > 0):
				sum_wp = np.zeros((nd, 1))
				for w_j in W:
					sum_wp += w_p.T.dot(w_j) * (w_j)
				w_p = w_p - sum_wp
			w_p = w_p / np.linalg.norm(w_p, ord=2)
			TERMINATE = not check_diff(w_p, w_old)
		W.append(w_p)
	W = np.asarray(W)
	W = W.squeeze()

	S = X.dot(W)
	return W, S

def check_diff(w_p, w_old, threshold=0.999):
	IS_DIFF = True
	cond = w_p.T.dot(w_old)
	print(cond)
	if cond > threshold:
		IS_DIFF = False
	return IS_DIFF

def weight_init(nd, init_mode='random'):
	if init_mode == 'random':
		theta = np.random.randn(nd, 1) 
	elif init_mode=='xavier':
		"""
		Glorot, Xavier, and Yoshua Bengio. 
		"Understanding the difficulty of training deep feedforward neural networks." 
		Proceedings of the thirteenth international conference on artificial 
		intelligence and statistics. 2010.
		"""
		pass
	else:
		pass
	return theta

def main():
	# generate data https://scikit-learn.org/stable/auto_examples/decomposition/plot_ica_blind_source_separation.html
	np.random.seed(0)
	n_samples = 2000
	from scipy import signal
	time = np.linspace(0, 8, n_samples)

	s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
	s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
	s3 = signal.sawtooth(2 * np.pi * time)  # Signal 3: saw tooth signal

	S = np.c_[s1, s2, s3]
	S += 0.2 * np.random.normal(size=S.shape)  # Add noise

	S /= S.std(axis=0)  # Standardize data
	# Mix data
	A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix
	X = np.dot(S, A.T)  # Generate observations

	# timer
	from time import time
	t0 = time()
	# signal source separation
	W, S_ = fast_ica(X)
	t1 = time()
	print('[Info] Fast ICA done in {:.2g} sec.'.format(t1 - t0))

	# plot
	plt.figure()

	models = [X, S, S_]
	names = ['Observations (mixed signal)',
			 'True Sources',
			 'ICA recovered signals']
	colors = ['red', 'steelblue', 'orange']

	for ii, (model, name) in enumerate(zip(models, names), 1):
		plt.subplot(3, 1, ii)
		plt.title(name)
		for sig, color in zip(model.T, colors):
			plt.plot(sig, color=color)

	plt.tight_layout()
	plt.show()
	input("Press Enter to continue...")
	
	return

if __name__ == '__main__':
	main()
