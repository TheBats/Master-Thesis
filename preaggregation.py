import torch

def nnm(to_aggregate, n_minus_f):
	new = []

	distances = torch.zeros((len(to_aggregate[0]), len(to_aggregate[0]))).cuda()
	for idx, m in enumerate(to_aggregate):
		for sub_idx, sub_m in enumerate(m):
			for sub_sub_idx, sub_sub_m in enumerate(m):
				distances[sub_idx, sub_sub_idx] += torch.norm((sub_sub_m-sub_m))

	sorted_idx = torch.argsort(distances, 1)
	
	for worker, idxs in zip(range(len(to_aggregate[0])), sorted_idx):
		for idx, m in enumerate(to_aggregate):
			if worker == 0:
				new.append(torch.unsqueeze(torch.mean(m[idxs[:n_minus_f]], 0), 0))
			else:
				new[idx] = torch.cat((new[idx], torch.unsqueeze(torch.mean(m[idxs[:n_minus_f]], 0), 0)))

	'''
	for m in moment1:
		moments1.append(torch.unsqueeze(m, 0))

	if get_two_moments:
		for m in moment2:
			moments2.append(torch.unsqueeze(m, 0))

	for idx, m in enumerate(moment1):
		moments1[idx] = torch.cat((moments1[idx], torch.unsqueeze(m, 0)))

	for idx, m in enumerate(to_aggregate):
		distances = torch.zeros((m.size()[0], m.size()[0]))
		for sub_idx, sub_m in enumerate(m):
			for sub_sub_idx, sub_sub_m in enumerate(m):
				distances[sub_idx, sub_sub_idx] = torch.norm((sub_sub_m-sub_m))

		sort_idx = torch.argsort(distances)

		for i, r in enumerate(sort_idx):
			tmp = torch.mean(m[r[:n_minus_f]], 0)

			if len(new) == idx:
				new.append(torch.unsqueeze(tmp, 0))
			else:
				new[idx] = torch.cat((new[idx], torch.unsqueeze(tmp, 0)))
	'''			
	return new