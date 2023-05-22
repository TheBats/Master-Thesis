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

def get_test_set_mnist(seed=1):
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


def get_test_set_cifar10(seed=1):
	print("Get test CIFAR10")
	mnist_test_set = torchvision.datasets.CIFAR10(root='.', train=False,
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


def read_homogenous_data(number_participants, dset_name):
    print(f"Reading: datasets/{dset_name}/homogenous/{number_participants} ")
    files = os.listdir(f"datasets/{dset_name}/homogenous/{number_participants}")
    datasets = []

    for file in files:
        datasets.append(torch.load(f"datasets/{dset_name}/homogenous/{number_participants}/{file}"))

    return datasets


def read_heterogenous_data(number_participants, alpha, dset_name):
	print(f"Reading: datasets/{dset_name}/heterogenous/{number_participants} ")

	files = os.listdir(f"datasets/{dset_name}/heterogenous/{number_participants}_{alpha}")
	datasets = []

	for file in files:
		datasets.append(torch.load(f"datasets/{dset_name}/heterogenous/{number_participants}_{alpha}/{file}"))

	return datasets


def homogenous_data_preparation(dataset="MNIST", number_participants=1):
    
    if dataset == "MNIST":
    	mnistTrainSet = torchvision.datasets.MNIST(root='.', train=True,
                                               download=True, transform=transformations)
    else:
    	trainset = torchvision.datasets.CIFAR10(root='./cifar10', train=True,
                                        download=True, transform=transform)
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
	transformations = transforms.Compose([transforms.ToTensor()])
	data = datasets.MNIST(root='.', download=True, transform=transformations)

	alphas = torch.tensor([[alpha for _ in range(10)] for _ in range(number_participants)])
	concentrations = torch.distributions.dirichlet.Dirichlet(alphas).sample()

	mnist_train_set = torchvision.datasets.MNIST(root='.', train=True,
	                                           download=True, transform=transformations)

	total_label_counts = torch.zeros(10)
	alphas = torch.tensor([alpha for _ in range(10)])
	data_bucket = [[] for _ in range(10)]

	count = torch.zeros(10)

	for (d, l) in mnist_train_set:
	    total_label_counts[l] += 1
	    data_bucket[l].append((d, l))
	    count[l] += 1

	alphas = torch.tensor([[alpha for _ in range(10)] for _ in range(number_participants)])

	torch.manual_seed(1)
	concentrations = torch.distributions.dirichlet.Dirichlet(alphas).sample()

	total_col = torch.zeros(10)
	total_row = torch.zeros(15)

	denom = torch.sum(concentrations, 0)

	for idx, row in enumerate(concentrations):
	    concentrations[idx, :] = row/denom

	final_sets = []

	total_counts = torch.zeros(10)

	for n, con in enumerate(concentrations):
	    tmp = []
	    con = torch.round(con*torch.min(count))
	    	    
	    for idx, i in enumerate(con):
	        for _ in range(int(i)):
	            # tmp.append(data_bucket[idx].pop())
	            total_counts[idx] += 1
	            
	    tmp_mnist_train_loader = torch.utils.data.DataLoader(tmp, batch_size=25,
														shuffle=True, num_workers=2)
	torch.save(tmp_mnist_train_loader, f"datasets/heterogenous/{number_participants}_{alpha}/{n}")
	print(torch.sum(total_counts))

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

	for x in ["results/mnist/homogenous", "results/cifar10/heterogenous", "results/cifar10/homogenous", "results/cifar10/heterogenous"]: #test_accuracy", "results/train_loss"]:
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