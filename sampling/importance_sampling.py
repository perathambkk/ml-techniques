"""
Author: Peratham Wiriyathammabhum


"""
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy import stats
eps = np.finfo(float).eps

"""
Importance sampling is a framework. It enables estimation of an r.v. X from a black box target distribution f 
using a known proposal distribution g (importance distribution) representing an r.v. Y. 
Well, we cannot sample from f but we can evaluate using f.

That is, we sample Y from g to estimate probabilities of X in f.
We multiply some function h(x) with the probability value for each sample using the importance weight f/g.

See: http://www.acme.byu.edu/wp-content/uploads/2016/12/Vol1B-MonteCarlo2-2017.pdf
	https://machinelearning1.wordpress.com/2017/10/22/importance-sampling-a-tutorial/
"""
def estimate_p_gt_3_for_gaussian(nsamples=2000):
	"""
	 The answer should approach 0.0013499 for sufficiently large samples.
	 See: http://www.acme.byu.edu/wp-content/uploads/2016/12/Vol1B-MonteCarlo2-2017.pdf
	"""
	h = lambda x: x > 3
	f = lambda x: stats.norm().pdf(x)
	g = lambda x: stats.norm(loc=4, scale=1).pdf(x)

	X = np.random.normal(4, scale=1, size=nsamples) # samples from g

	est = np.sum(h(X) * (f(X) / g(X)))
	est /= nsamples
	return est

def main(opts):
	nsamples = opts['nsamples']
	est = estimate_p_gt_3_for_gaussian(nsamples=nsamples)
	print('[Info] estimate P(X > 3)=0.0013499 for normal distribution: {:.7f}'.format(est))
	print('[Info] estimation difffer by: {:.7f}'.format(np.absolute(est - 0.0013499)))
	return

if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser(description='run importance sampling.')

	parser.add_argument('--num_samples', dest='nsamples',
					  help='number of samples',
					  default=2000, type=int)
	args = parser.parse_args()
	opts = vars(args)

	main(opts)
