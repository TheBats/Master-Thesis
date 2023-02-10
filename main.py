import copy
import os
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

save = False
number_workers = 15
rounds = 500
f = 5 # Number of attackers 

number_moments = 1
get_two_moments = False

print(get_two_moments)

lr = 0.001 if number_moments == 2 else 0.75
beta = (0.9, 0.999)
eps = 10e-8

loss = nn.CrossEntropyLoss

test_loader = get_test_set()
data = read_data(number_workers)
worker = []

for attack in ["little"]: #,"sign_flip", "label_flip"]:
	for aggregation_method in ["mda"]: #, "avg", "cwm", "meamed", "cwtm"]:
		torch.manual_seed(seed)
		model = CnnMist()

		keys = list(model.state_dict().keys())
		print(f"Attack: {attack} Aggregation: {aggregation_method}")

		for idx, d in enumerate(data):
			local_model = copy.deepcopy(model)
			new_worker = Worker(dataset=d, model=local_model, loss=loss, number_moments=number_moments,
								attack = attack if idx < f and attack in ["sign_flip", "label_flip"] else "None",
								beta = beta, get_two_moments=get_two_moments)

			worker.append(new_worker)

		losses = np.zeros([number_workers, rounds])
		accuracies = np.zeros([1, rounds])

		xs = np.arange(0, rounds)

		for r in range(rounds):
			moments1 = []
			moments2 = []

			# accuracies[0, r] = worker[0].test(test_loader)

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
						aggregated2 = aggregation.mda(moments1, number_workers-f)

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

		if number_moments == 1:
			name = "1_moment"
		elif get_two_moments == True:
			name = "2_2_moments"
		else:
			name = "2_moments"

		if save:
			with open(f'results/train_loss/{name}/{attack}/{aggregation_method}_{number_workers}_{f}_{rounds}.npy', 'wb+') as d:
				np.save(d, losses)
				d.flush()
				os.fsync(d.fileno())

			with open(f'results/test_accuracy/{name}/{attack}/{aggregation_method}_{number_workers}_{f}_{rounds}.npy', 'wb+') as d:
				np.save(d, accuracies)
				d.flush()
				os.fsync(d.fileno())

		fig = go.Figure()
		for w in range(number_workers):
			fig.add_trace(go.Scatter(x=xs, y=losses[w,:], mode='lines', name=f'Model {w}'))
		fig.show()