import copy
import os
import torch
import dash
import math

import torch.nn as nn
import numpy as np
import plotly.graph_objects as go
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from model import CnnMist, CnnCifar10
from data_preparation import read_homogenous_data, read_heterogenous_data, get_test_set_cifar10, get_test_set_mnist
from worker import Worker

import aggregation
import attacks
import preaggregation

import random
import numpy as np

# Seed
seed = 123
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Homo: Cross Entropy, n=15 workers, constant learning rate of 0.75, clipping 2
# Heter: NLLn, n=17, decay learning rate starting at 0.75, momentum 0.9

save = True
plot = False
heter = True

dset_name = "cifar10"
pre_ag = "none" if not heter else "nnm"
alpha = 0.1
number_workers = 15
rounds = 1500
f = 4 # Number of attackers 

eps = 10e-8
weight_decay = 10e-4

if heter:
	loss = nn.NLLLoss()
else:
	loss = nn.CrossEntropyLoss()

if dset_name == "mnist":
	test_loader = get_test_set_mnist()
if dset_name == "cifar10": 
	test_loader = get_test_set_cifar10()

data = read_heterogenous_data(number_workers, alpha, dset_name) if heter else read_homogenous_data(number_workers, dset_name)

'''
if dset_name == "mnist":
	data = read_heterogenous_data(number_workers, alpha) if heter else read_homogenous_data(15)
if dset_name == "cifar10":
	data = read_homogenous_cifar(number_workers)
'''

for opt in [0, 1]:
	match opt:
		case 0: 
			print("-- No moment --")
			number_moments = 1
			get_two_moments = False
			beta = (0, 0)
		case 1: 
			print("-- 1 moment --")
			number_moments = 1
			get_two_moments = False
			beta = (0.9, 0)
		case 2: 
			print("-- 2 moments --")
			number_moments = 2
			get_two_moments = False
			beta = (0.99, 0.999)
		case 3: 
			print("-- 2 2 moments --")
			number_moments = 2
			get_two_moments = True
			beta = (0.99, 0.999)

	if number_moments == 2:
		max_lr = 0.001
		lr = 0.001
	else:
		if dset_name == "mnist":
			max_lr = 0.75
			lr = 0.75

		if dset_name == "cifar10":
			max_lr = 0.25
			lr = 0.25

	for attack in ["sign_flip", "label_flip", "inf", "-inf", "empire", "little"] if opt == 2 else ["none", "sign_flip", "label_flip", "inf", "-inf", "empire", "little"]: #"none", "little", 
		for a, aggregation_method in enumerate(["mda", "cwtm", "cwm", "avg", "meamed"]): #["cwtm"]): 

			worker = []
			if dset_name == "mnist":
				torch.manual_seed(seed)
				model = CnnMist()

			if dset_name == "cifar10":
				torch.manual_seed(seed)
				model = CnnCifar10()

			print(f"Attack: {attack} Pre-aggregation: {pre_ag} Aggregation: {aggregation_method}")

			for idx, d in enumerate(data):

				if dset_name == "mnist":
					torch.manual_seed(seed)
					local_model = CnnMist()

				if dset_name == "cifar10":
					torch.manual_seed(seed)
					local_model = CnnCifar10()

				new_worker = Worker(dataset=d, model=local_model, loss=loss, number_moments=number_moments,
									attack = attack if idx < f and attack in ["sign_flip", "label_flip", "-inf", "inf"] else "none",
									beta = beta, get_two_moments=get_two_moments, weight_decay=weight_decay)

				worker.append(new_worker)

			keys = list(worker[1].model.state_dict().keys())
			
			losses = np.zeros([number_workers, rounds])
			accuracies = np.zeros([1, rounds])

			xs = np.arange(0, rounds)

			for r in range(rounds):
				moments1 = []
				moments2 = []

				accuracies[0, r] = worker[0].test(test_loader)

				for w in range(number_workers):
					if get_two_moments:
						loss, moment1, moment2 = worker[w].take_step()
					else:
						loss, moment1  = worker[w].take_step()

					losses[w, r] = loss.cpu().detach()

					if moments1 == []:
						for m in moment1:
							moments1.append(torch.unsqueeze(m, 0))

						if get_two_moments:
							for m in moment2:
								moments2.append(torch.unsqueeze(m, 0))
					else:
						for idx, m in enumerate(moment1):
							moments1[idx] = torch.cat((moments1[idx], torch.unsqueeze(m, 0)))

						if get_two_moments:
							for idx, m in enumerate(moment2):
								moments2[idx] = torch.cat((moments2[idx], torch.unsqueeze(m, 0)))

				match attack:
					case "little":
						attacks.little(moments1, number_workers, f)
						if get_two_moments:
							attacks.little(moments2, number_workers, f)

					case "empire":
						attacks.empire(moments1, f)
						if get_two_moments:
							attacks.little(moments2, number_workers, f)

				match pre_ag:
					case "nnm":
						moments1 = preaggregation.nnm(moments1, number_workers-f)
						if get_two_moments:
							moments2 = preaggregation.nnm(moments2, number_workers-f)

				match aggregation_method:
					case "avg":
						aggregated = aggregation.avg(moments1)
						if get_two_moments:
							aggregated2 = aggregation.avg(moments2)

					case "cwtm":
						aggregated = aggregation.cwtm(moments1, number_workers, f)
						if get_two_moments:
							aggregated2 = aggregation.cwtm(moments2, number_workers, f)

					case "meamed":
						aggregated = aggregation.meamed(moments1, number_workers-f)
						if get_two_moments:
							aggregated2 = aggregation.meamed(moments2, number_workers-f)

					case "cwm":
						aggregated = aggregation.cwm(moments1)
						if get_two_moments:
							aggregated2 = aggregation.cwm(moments2)

					case "mda":
						aggregated = aggregation.mda(moments1, number_workers-f)
						if get_two_moments:
							aggregated2 = aggregation.mda(moments2, number_workers-f)

				if get_two_moments and number_moments == 2:
					# Aggregate the momentums
					sqrt = torch._foreach_sqrt(aggregated2)
					torch._foreach_add_(sqrt, eps)
					aggregated = torch._foreach_div(aggregated, sqrt)

				new_state = {}
				
				for idx, (params, moments) in enumerate(zip(model.parameters(), aggregated)):
					new_state[keys[idx]] = params - lr * moments.cpu().detach()
				
				model.load_state_dict(new_state)

				for w in worker:
					w.model.load_state_dict(model.state_dict())

				if dset_name == "mnist" and number_moments == 1 and heter:
					lr = max_lr/(1+ math.floor(r/50))

				if dset_name == "cifar10" and number_moments == 1 and heter and r == 1500:
					lr = max_lr/(1+ math.floor(r/50))

				#if number_moments == 2:
				#	lr = max_lr*math.sqrt(1-beta[1]**(r+1))/(1-beta[0]**(r+1))

			if beta[0] == 0:
				name = "no_moment"
			elif number_moments == 1:
				name = "1_moment"
			elif get_two_moments == True:
				name = "2_2_moments"
			else:
				name = "2_moments"

			if heter:
				sub = f"{dset_name}/heterogenous"
				f_name = f"{pre_ag}_{alpha}_{aggregation_method}_{number_workers}_{f}_{rounds}"
			else:
				sub = f"{dset_name}/homogenous"
				f_name = f"{aggregation_method}_{number_workers}_{f}_{rounds}"

			print(f'results/{sub}/train_loss/{name}/{attack}/{f_name}.npy')

			if save:
				with open(f'results/{sub}/train_loss/{name}/{attack}/{f_name}.npy', 'wb+') as d:
					np.save(d, losses)
					d.flush()
					os.fsync(d.fileno())
				d.close()

				with open(f'results/{sub}/test_accuracy/{name}/{attack}/{f_name}.npy', 'wb+') as d:
					np.save(d, accuracies)
					d.flush()
					os.fsync(d.fileno())
				d.close()
				
			if plot:
				fig = make_subplots(rows=1, cols=2, column_titles = [f"Training Loss: {aggregation_method}", f"Accuracy: {aggregation_method}"])
				for w in range(number_workers):
					fig.add_trace(go.Scatter(x=xs, y=losses[w,:], mode='lines', name=f'Model {w}'), row=1, col=1)

				fig.add_trace(
					go.Scatter(
						name = "",
						x = xs,
						y = accuracies[0],
						mode = 'lines',
						legendgroup = "g1",
						showlegend= False,
						line_color='#9274cf'
					), row=1, col=2
				)

				fig.show()
				'''
				fig = go.Figure()
				for w in range(number_workers):
					fig.add_trace(go.Scatter(x=xs, y=losses[w,:], mode='lines', name=f'Model {w}'))
				fig.show()
				'''