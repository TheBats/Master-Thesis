import copy
import torch
import dash

import torch.nn as nn
import numpy as np
import plotly.graph_objects as go

from model import CnnMist
from data_preparation import read_data, get_test_set
from worker import Worker
import aggregation
import attacks

save = True
number_workers = 15
rounds = 500
f = 5 # Number of attackers 
aggregation_method = "avg" # "cwtm", "meamed", "avg", cwm
attack = "little" # little, sign_flip, label_flip

number_moments = 2
lr = 0.001 if number_workers == 1 else 0.75
beta = (0.9, 0.999)

model = CnnMist()
loss = nn.CrossEntropyLoss()

test_loader = get_test_set()

keys = list(model.state_dict().keys())
data = read_data(number_workers)
worker = []

for idx, d in enumerate(data):
	local_model = copy.deepcopy(model)
	new_worker = Worker(dataset=d, model=local_model, loss=loss, 
						attack = attack if idx < f and attack in ["sign_flip", "label_flip"] else "None",
						beta = beta)
	worker.append(new_worker)

losses = np.zeros([number_workers, rounds])
accuracies = np.zeros([1, rounds])

xs = np.arange(0, rounds)

for r in range(rounds):
	moments = []
	accuracies[0, r] = worker[0].test(test_loader)
	for w in range(number_workers):
		loss, moment = worker[w].take_step()
		losses[w, r] = loss.cpu().detach()

		if moments == []:
			for m in moment:
				moments.append(torch.unsqueeze(m, 0))
		else:
			for idx, m in enumerate(moment):
				moments[idx] = torch.cat((moments[idx], torch.unsqueeze(m, 0)))
	
	match attack:
		case "little":
			attacks.little(moments, number_workers, f)

	match aggregation_method:
		case "avg":
			aggregated = aggregation.avg(moments)
		case "cwtm":
			aggregated = aggregation.cwtm(moments, number_workers, f)
		case "meamed":
			aggregated = aggregation.meamed(moments, number_workers-f)
		case "cwm":
			aggregated = aggregation.cwm(moments)

	new_state = {}
	for idx, (params, moments) in enumerate(zip(model.parameters(), aggregated)):
		new_state[keys[idx]] = params - lr * moments.cpu().detach()
	
	model.load_state_dict(new_state)

	for w in worker:
		w.model.load_state_dict(model.state_dict())

name = "1_moment" if number_moments == 1 else "2_moments"

if save:
	np.save(f"results/train_loss/{name}/{attack}/{aggregation_method}_{number_workers}_{f}_{rounds}_{beta}", losses)
	np.save(f"results/test_accuracy/{name}/{attack}/{aggregation_method}_{number_workers}_{f}_{rounds}_{beta}", accuracies)

fig = go.Figure()
for w in range(number_workers):
	fig.add_trace(go.Scatter(x=xs, y=losses[w,:], mode='lines', name=f'Model {w}'))

fig.show()