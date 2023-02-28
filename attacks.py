import torch

from scipy.stats import norm

def little(moments, num_workers, f):
	# Required corrupted workers for majority
	s = (num_workers / 2) + 1 - f
	p = (num_workers-s)/num_workers
	z = norm.ppf(p)

	for idx, m in enumerate(moments):
		# mean = torch.mean(m[:f], dim=0)
		std = torch.std(m[:f], dim=0)
		# x =  mean + z * std

		for i in range(f):
			m[i] = m[i] - std


def empire(moments, f):
	for idx, m in enumerate(moments):
		for i in range(f):
			m[i] = -0.9*m[i]

def inf(moments, f, inf):
	for idx, m in enumerate(moments):
		for i in range(f):
			m[i] = inf[idx]

	print(moments)