import math
import os
import torch
import torchvision
import numpy as np

from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedShuffleSplit

transformations = transforms.Compose([transforms.ToTensor()])
data = datasets.MNIST(root='.', download=True, transform=transformations)

def get_test_set():
	mnistTestSet = torchvision.datasets.MNIST(root='.', train=False,
                                              download=True, transform=transformations)
	testloader = torch.utils.data.DataLoader(mnistTestSet, batch_size=25,
                                         	 shuffle=False, num_workers=2)

	return testloader

def read_data(number_participants):
    files = os.listdir(f"datasets/{number_participants}")
    datasets = []

    for file in files:
        datasets.append(torch.load(f"datasets/{number_participants}/{file}"))

    return datasets


def custom_data_splitting(number_participants=1):
    mnistTrainSet = torchvision.datasets.MNIST(root='.', train=True,
                                               download=True, transform=transformations)
    min_occuring = 5421 # 5
    ratio = math.floor(min_occuring/number_participants)

    if not os.path.isdir("datasets"):
    	os.makedirs("datasets")

    if not os.path.isdir(f"datasets/{number_participants}"):
    	os.makedirs(f"datasets/{number_participants}")

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
        torch.save(tmp_mnistTrainLoader, f"datasets/{number_participants}/{idx}")

    return train_loaders

def build_directories():

	if not os.path.isdir("results"):
		os.makedirs("results")
	if not os.path.isdir("results/train_loss"):
		os.makedirs("results/train_loss")
	if not os.path.isdir("results/test_accuracy"):
		os.makedirs("results/test_accuracy")

	for p in ["results/test_accuracy", "results/train_loss"]:
		if not os.path.isdir(f"{p}/1_moment"):
			os.makedirs(f"{p}/1_moment")
		if not os.path.isdir(f"{p}/2_moments"):
			os.makedirs(f"{p}/2_moments")

		for attack in ["little", "sign_flip", "label_flip"]:
			if not os.path.isdir(f"{p}/1_moment/{attack}"):
				os.makedirs(f"{p}/1_moment/{attack}")
			if not os.path.isdir(f"{p}/2_moments/{attack}"):
				os.makedirs(f"{p}/2_moments/{attack}")