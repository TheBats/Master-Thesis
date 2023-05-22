import torch

import numpy as np

# Average of all 
def avg(to_aggregate):
	for idx, m in enumerate(to_aggregate):
		to_aggregate[idx] = torch.mean(m, 0)

	return to_aggregate


# Minimum Diameter Averaging
# 1. Choose a set of minimum cardinality of n-f items with smallest diameter
# To Do: Speed up by reusing distances
def mda(to_aggregate, n_minus_f):
	distances = np.zeros((len(to_aggregate[0]), len(to_aggregate[0])))
	for idx, m in enumerate(to_aggregate):
		for sub_idx, sub_m in enumerate(m):
			for sub_sub_idx, sub_sub_m in enumerate(m):
				distances[sub_idx, sub_sub_idx] += torch.norm((sub_sub_m-sub_m))

	arg_sorted_distances = np.argsort(distances, 1)
	idx_largest_per_row = arg_sorted_distances[:, -1]
	values_largest_score = [distances[row, col] for (row, col) in enumerate(idx_largest_per_row)]
	final_idx = np.argsort(values_largest_score)[:n_minus_f]

	for idx, m in enumerate(to_aggregate):
		to_aggregate[idx] = torch.mean(m[final_idx], 0)

	return to_aggregate

# Coordinate-Wise Trimmed Mean
# 1. Sort the kth coordinate of the input vector in ascending order
# 2. Sum [f+1, n-f]
# 3. Divide by n-2f
def cwtm(to_aggregate, n, f):
	for idx, m in enumerate(to_aggregate):
		sorted_m, _ = torch.sort(m, 0) 
		to_aggregate[idx] = torch.mean(sorted_m[range(f+1,n-f)], 0)

	return to_aggregate


# Mean around Median
# 1. Compute the median
# 2. Compute the coordinate wise distance between the median and entry
# 3. Average the n-f entries closest to the median
def meamed(to_aggregate, n_minus_f):
	for idx, m in enumerate(to_aggregate):
		median = torch.median(m, 0)[0]
		dist = torch.sqrt(torch.pow(m-median, 2))
		_, sorted_idx = torch.sort(dist, 0)
		sorted_m = torch.gather(m, 0, sorted_idx)
		to_aggregate[idx] = torch.mean(sorted_m[:n_minus_f], 0)

	return to_aggregate


# Coordinate-Wise Median
def cwm(to_aggregate):
	for idx, m in enumerate(to_aggregate):
		to_aggregate[idx] = torch.median(m, 0)[0]

	return to_aggregate

#data = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[5.0, 1.0, 1.0], [2.5, 5., 2.0]], [[10.0, 100.0, 10.0], [10.0, 100.0, 10.0]], [[5.0, 0.0, 1.0], [5.0, 0.0, 1.0]]])
#print(data.size())
#print(mda([data], 2))