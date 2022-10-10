"""
Author: Peratham Wiriyathammabhum


"""
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy import stats
from tqdm import tqdm
eps = np.finfo(float).eps

"""
Problem 2. A tech support hotline receives an average of 2 calls per minute.
What is the probability that they will have to wait at least 10 minutes
to receive 9 calls?

See: http://www.acme.byu.edu/wp-content/uploads/2016/12/Vol1B-MonteCarlo2-2017.pdf
	https://machinelearning1.wordpress.com/2017/10/22/importance-sampling-a-tutorial/
"""
def estimate_gt10_a9_for_gamma():
	"""
	 The answer should approach 0.00208726 for sufficiently large samples.
	 See: https://web.archive.org/web/20200822153616/http://www.acme.byu.edu/wp-content/uploads/2016/12/Vol1B-MonteCarlo2-2017.pdf
	"""
	# monte carlo integration
	h = lambda x : x > 10

	print('[Info] Monte Carlo integration estimate...')
	MC_estimates = []
	for N in tqdm(range(5000,505000,5000)):
		X = np.random.gamma(9,scale=0.5,size=N)
		MC = 1./N*np.sum(h(X))
		MC_estimates.append(MC)
	MC_estimates = np.array(MC_estimates)

	# importance sampling
	f = lambda x: stats.gamma(a=9, scale=0.5).pdf(x)
	# g = lambda x: 1.-stats.gamma(a=4,scale=0.5).pdf(x)
	g = lambda x: stats.gamma(a=4, scale=1).pdf(x)
	# g = lambda x: stats.norm(loc=4, scale=1).pdf(x)

	print('[Info] Importance Sampling estimate...')
	IS_estimates = []
	for N in tqdm(range(5000,505000,5000)):
		# X = np.random.normal(4, scale=1, size=N)
		X = np.random.gamma(4,scale=1,size=N)
		est = np.sum(h(X) * (f(X) / g(X)))
		# est /= N
		est /= np.sum((f(X) / g(X)))
		IS_estimates.append(est)
	IS_estimates = np.array(IS_estimates)
	return MC_estimates, IS_estimates

def main(opts):
	nsamples = opts['nsamples']
	m_est, i_est = estimate_gt10_a9_for_gamma()

	est_target = 1 - stats.gamma(a=9,scale=0.5).cdf(10)

	fig = plt.figure()
	# plt.subplot(211)
	# plt.plot(list(range(5000,505000,5000)), m_est)

	# plt.subplot(212)
	# plt.plot(list(range(5000,505000,5000)), i_est)

	plt.plot(list(range(5000,505000,5000)), np.absolute(m_est-est_target),list(range(5000,505000,5000)), np.absolute(i_est-est_target))
	plt.legend(np.array(["MC_estimate_errors","IS_estimates_errors"]))
	plt.show()
	input("Press Enter to continue...")

	print('[Info] monte carlo integration estimate p(gamma(a=9)>10)=0.00208726 for gamma distribution: {:.7f}'.format(m_est[-1]))
	print('[Info] monte carlo integration estimation difffer by: {:.8f}'.format(np.absolute(m_est[-1] - 0.00208726)))
	print('[Info] importance sampling integration estimate p(gamma(a=9)>10)=0.00208726 for gamma distribution: {:.7f}'.format(i_est[-1]))
	print('[Info] importance sampling integration estimation difffer by: {:.8f}'.format(np.absolute(i_est[-1] - 0.00208726)))
	print('[Info] actual prob. {:.8f} from 1 - stats.gamma(a=9,scale=0.5).cdf(10)'.format(1 - stats.gamma(a=9,scale=0.5).cdf(10)))
	return

if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser(description='run monte carlo integration.')

	parser.add_argument('--num_samples', dest='nsamples',
					  help='number of samples',
					  default=5000, type=int)
	args = parser.parse_args()
	opts = vars(args)

	main(opts)
