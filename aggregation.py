import torch

import numpy as np

# Average of all 
def avg(to_aggregate):
	for idx, m in enumerate(to_aggregate):
		to_aggregate[idx] = torch.mean(m, 0)

	return to_aggregate


# Minimum Diameter Averaging
# 1. Choose a set of minimum cardinality of n-f items with smallest diameter
def mda(to_aggregate, n_minus_f):
	for idx, m in enumerate(to_aggregate):
		distances = np.zeros((m.size()[0], m.size()[0]))
		for sub_idx, sub_m in enumerate(m):
			for sub_sub_idx, sub_sub_m in enumerate(m):
				distances[sub_idx, sub_sub_idx] = torch.norm((sub_sub_m-sub_m)) # torch.linalg.norm((sub_sub_m-sub_m), dim=1).sum(dim=1).sum(dim=1).sum(dim=1)

		sum_dist = np.sum(distances, 1)
		sort_idx = np.argsort(sum_dist)

		to_aggregate[idx] = torch.mean(m[sort_idx[:n_minus_f]], 0)

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

# Geometric Median
# 1. Find a vector minimizing the difference between it and all other vectors
def gm(to_aggregate):
	pass

# Krum
def krum(to_aggregate, n_minus_f_minus_1, q):
    distances = None

    for i in range(0, to_aggregate.shape[0]):
        new_row = np.linalg.norm(to_aggregate-to_aggregate[i], axis=1)

        if distances is None:
            distances = new_row.reshape(1, len(to_aggregate))
        else:
            distances = np.append(distances, [new_row], axis=0)

    sorted_scores = np.argsort(distances, axis=1)
    selection = sorted_scores[:, 1:n_minus_f_minus_1+1]
    print(selection)

    if q > 1 and n_minus_f_minus_1 > 1:
        print("In")
        pass
    else:
        print(f"To aggregate: {to_aggregate[selection]}")
        return to_aggregate[selection]

# print(mda([torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[5.0, 1.0, 1.0], [5.0, 1.0, 1.0]], [[5.0, 1.0, 1.0], [2.5, 10, 11]]])], 3))