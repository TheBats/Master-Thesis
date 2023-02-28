import math
import os
import torch
import torchvision
import numpy as np

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from collections import Counter

transformations = transforms.Compose([transforms.ToTensor()])
data = datasets.MNIST(root='.', download=True, transform=transformations)

def get_test_set(seed=1):
	mnist_test_set = torchvision.datasets.MNIST(root='.', train=False,
                                              download=True, transform=transformations)

	_, test_indices, _, _ = train_test_split(
		range(len(mnist_test_set)),
		mnist_test_set.targets,
		stratify=mnist_test_set.targets,
		test_size=2000,
		random_state=seed
	)

	final_test = Subset(mnist_test_set, test_indices)
	testloader = DataLoader(final_test, batch_size=25,
                            shuffle=False, num_workers=2)

	return testloader


def read_homogenous_data(number_participants):
    files = os.listdir(f"datasets/homogenous/{number_participants}")
    datasets = []

    for file in files:
        datasets.append(torch.load(f"datasets/homogenous/{number_participants}/{file}"))

    return datasets


def read_heterogenous_data(number_participants, alpha):
	files = os.listdir(f"datasets/heterogenous/{number_participants}_alpha_{alpha}")
	datasets = []

	for file in files:
		datasets.append(torch.load(f"datasets/heterogenous/{number_participants}_alpha_{alpha}/{file}"))

	return datasets


def homogenous_data_preparation(number_participants=1):
    mnistTrainSet = torchvision.datasets.MNIST(root='.', train=True,
                                               download=True, transform=transformations)
    min_occuring = 5421 # 5
    ratio = math.floor(min_occuring/number_participants)

    if not os.path.isdir("datasets"):
    	os.makedirs("datasets")

    if not os.path.isdir("datasets/homogenous"):
    	os.makedirs("datasets/homogenous")

    if not os.path.isdir(f"datasets/homogenous/{number_participants}"):
    	os.makedirs(f"datasets/homogenous/{number_participants}")

    labels = None
    for (d, l) in mnistTrainSet:
        if labels is None:
            labels = np.array([l])
        else:
            labels = np.concatenate((labels, [l]), axis=0) # .append(l)

    all_idx = np.arange(0, len(labels))
    splits = [np.array([]) for _ in range(number_participants)]
    new = [[] for _ in range(number_participants)]

    for i in range(0, 10):
        idx = all_idx[np.where(labels == i)][:ratio*number_participants]
        idx = np.array_split(idx, number_participants)
        
        for x in range(0, number_participants):
            for sp in idx[x]:
                new[x].append(data[sp])
                
    train_loaders = []
    for idx, split in enumerate(new):
        tmp_mnistTrainLoader = torch.utils.data.DataLoader(split, batch_size=25,
                                                           shuffle=True, num_workers=2)
        train_loaders.append(tmp_mnistTrainLoader)
        torch.save(tmp_mnistTrainLoader, f"datasets/homogenous/{number_participants}/{idx}")

    return train_loaders


def heterogenous_data_preparation(number_participants=1, alpha=0.1, seed=1):
	alphas = torch.tensor([[alpha for _ in range(10)] for _ in range(number_participants)])
	concentrations = torch.distributions.dirichlet.Dirichlet(alphas).sample()

	mnist_train_set = torchvision.datasets.MNIST(root='.', train=True,
                                               download=True, transform=transformations)

	skf = StratifiedKFold(n_splits=number_participants, shuffle=True, random_state=seed)
	splits = skf.get_n_splits(range(len(mnist_train_set)), mnist_train_set.targets)

	labels = np.zeros((number_participants, round(len(mnist_train_set)/number_participants)))
	min_occ = 1000000

	for w, (_, test_indices) in enumerate(skf.split(range(len(mnist_train_set)), mnist_train_set.targets)):
		sub_set = Subset(mnist_train_set, test_indices)

		for idx, (_, label) in enumerate(sub_set):
			labels[w, idx] = label

		occurences = min(Counter(labels[w, :]).values())
		
		if occurences < min_occ:
			min_occ = occurences

	scaled_concentrations = torch.round(concentrations * min_occ)
	print(torch.sum(scaled_concentrations, dim=1))

	for idx, (_, test_indices) in enumerate(skf.split(range(len(mnist_train_set)), mnist_train_set.targets)):	
		sub_set = Subset(mnist_train_set, test_indices)

		final_idx = np.array([])

		new_dset = []

		for i, l in enumerate(np.unique(labels[idx, :])):
			to_select = labels[idx, :] == l 
			number = int(scaled_concentrations[idx,i].item())
			to_select = test_indices[to_select]
			
			if not to_select[:number] is []:
				#final_idx = np.concatenate((final_idx, to_select[:number]))
				for idxs in to_select[:number]:
					new_dset.append(mnist_train_set[int(idxs)])
		

		tmp_mnist_train_loader = torch.utils.data.DataLoader(new_dset, batch_size=25,
															shuffle=True, num_workers=2)

		torch.save(tmp_mnist_train_loader, f"datasets/heter/{number_participants}_alpha_{alpha}/{idx}")


def build_directories():
	if not os.path.isdir("results"):
		os.makedirs("results")
	if not os.path.isdir("results/homogenous"):
		os.makedirs("results/homogenous")
	if not os.path.isdir("results/heterogenous"):
		os.makedirs("results/heterogenous")


	'''
	if not os.path.isdir("results/train_loss"):
		os.makedirs("results/train_loss")
	if not os.path.isdir("results/test_accuracy"):
		os.makedirs("results/test_accuracy")
	'''
	for x in ["results/homogenous", "results/heterogenous"]: #test_accuracy", "results/train_loss"]:
		for i in ["test_accuracy", "train_loss"]:

			if not os.path.isdir(f"{x}/{i}"):
				os.makedirs(f"{x}/{i}")

			p = x + "/" + i 

			if not os.path.isdir(f"{p}/no_moment"):
				os.makedirs(f"{p}/no_moment")
			if not os.path.isdir(f"{p}/1_moment"):
				os.makedirs(f"{p}/1_moment")
			if not os.path.isdir(f"{p}/2_moments"):
				os.makedirs(f"{p}/2_moments")
			if not os.path.isdir(f"{p}/2_2_moments"):
				os.makedirs(f"{p}/2_2_moments")

			for attack in ["little", "sign_flip", "label_flip", "empire", "none", "-inf", "inf"]:
				if not os.path.isdir(f"{p}/1_moment/{attack}"):
					os.makedirs(f"{p}/1_moment/{attack}")
				if not os.path.isdir(f"{p}/2_moments/{attack}"):
					os.makedirs(f"{p}/2_moments/{attack}")
				if not os.path.isdir(f"{p}/2_2_moments/{attack}"):
					os.makedirs(f"{p}/2_2_moments/{attack}")
				if not os.path.isdir(f"{p}/no_moment/{attack}"):
					os.makedirs(f"{p}/no_moment/{attack}")

build_directories()