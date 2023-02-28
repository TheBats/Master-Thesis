import copy

import torch
import torch.nn as nn
import numpy as np

from torchvision import datasets
from torch import optim
from model import CnnMist

class Worker():
	
	def __init__(self, dataset: datasets, model: nn.Module, loss: nn, beta,
				 number_moments, get_two_moments, attack, eps=10e-8,
				 weight_decay=0):

		self.dataset = dataset
		self.model = model.to("cuda")
		self.loss = nn.CrossEntropyLoss()
		self.iterator = iter(dataset)
		self.attack = attack
		self.number_moments = number_moments
		self.get_two_moments = get_two_moments

		self.beta = beta
		self.eps = eps
		self.steps = 1
		self.weight_decay = weight_decay

		self.moment1 = []
		self.moment2 = []

		for i in self.model.parameters():

			if self.attack == "inf":
				self.moment1.append(torch.full_like(i, float('inf'), memory_format=torch.preserve_format))
				
				if self.number_moments == 2:
					self.moment2.append(torch.full_like(i, float('inf'),  memory_format=torch.preserve_format))
			
			elif self.attack == "-inf":
				self.moment1.append(torch.full_like(i, float('-inf'), memory_format=torch.preserve_format))
				
				if self.number_moments == 2:
					self.moment2.append(torch.full_like(i, float('-inf'), memory_format=torch.preserve_format))

			else:
				self.moment1.append(torch.zeros_like(i, memory_format=torch.preserve_format))
				
				if self.number_moments == 2:
					self.moment2.append(torch.zeros_like(i, memory_format=torch.preserve_format))

	def take_step(self):
		def _view_complex_as_real(tensor_list):
			return [torch.view_as_real(t) if torch.is_complex(t) else t for t in tensor_list]

		try:
			(data, label) = next(self.iterator)
		except:
			self.iterator = iter(self.dataset)
			(data, label) = next(self.iterator)

		data, label = data.to("cuda"), label.to("cuda")

		if self.attack == "label_flip":
			label = 9 - label

		pred = self.model(data)
		loss = self.loss(pred, label)

		self.model.zero_grad()
		loss.backward()

		if self.attack in ["inf", "-inf"]:
			if self.number_moments == 2:
				return loss, self.moment1, self.moment2

			return loss, self.moment1
		
		torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2)

		params = []
		grads = []
		for i in self.model.parameters():
			grads.append(i.grad)
			params.append(i)

		if self.weight_decay != 0:	
			torch._foreach_add_(grads, params, alpha=self.weight_decay)

		# First moment computation
		grads = _view_complex_as_real(grads)
		moment1 = _view_complex_as_real(self.moment1)

		torch._foreach_mul_(moment1, self.beta[0])
		torch._foreach_add_(moment1, grads, alpha= 1 - self.beta[0])

		if self.number_moments == 2:
			# Second moment computation
			moment2 = _view_complex_as_real(self.moment2)

			torch._foreach_mul_(moment2, self.beta[1])
			torch._foreach_addcmul_(moment2, grads, grads, value=1-self.beta[1])

		match self.attack:
			case "none" | "label_flip":
				if self.number_moments == 1:
					return (loss, self.moment1)

				else:
					# Correct the bias
					unbiased1 = torch._foreach_div(self.moment1, (1 - self.beta[0] ** self.steps))
					unbiased2 = torch._foreach_div(self.moment2, (1 - self.beta[1] ** self.steps))
					self.steps += 1 

					if self.get_two_moments:
						return (loss, unbiased1, unbiased2)
					else:
						sqrt = torch._foreach_sqrt(unbiased2) #unbiased2)
						torch._foreach_add_(sqrt, self.eps)
						new = torch._foreach_div(unbiased1, sqrt)
						return (loss, new)

			case "sign_flip":
				if self.number_moments == 1:
					flipped_moments = []
					[flipped_moments.append(-m) for m in self.moment1]
					return (loss, flipped_moments)
				else:

					# Correct the bias
					unbiased1 = torch._foreach_div(self.moment1, (1 - self.beta[0] ** self.steps))
					unbiased2 = torch._foreach_div(self.moment2, (1 - self.beta[1] ** self.steps))
					self.steps += 1 

					if self.get_two_moments:
						flipped_moments1 = []
						[flipped_moments1.append(-m) for m in unbiased1]

						flipped_moments2 = []
						[flipped_moments2.append(-m) for m in unbiased2]

						return (loss, flipped_moments1, flipped_moments2)

					else:
						sqrt = torch._foreach_sqrt(unbiased2) #unbiased2)
						torch._foreach_add_(sqrt, self.eps)
						new = torch._foreach_div(unbiased1, sqrt)

						flipped_moments = []
						[flipped_moments.append(-m) for m in new]
						return (loss, flipped_moments)


	def aggregate_momentums(self):
		# Aggregate the momentums
		sqrt = torch._foreach_sqrt(self.moment2) #unbiased2)
		torch._foreach_add_(sqrt, self.eps)
		new = torch._foreach_div(self.moment1, sqrt)

		return new


	def test(self, testset):
		correct = 0 
		total = 0
		with torch.no_grad():
			for data, labels in testset:
				data, labels = data.to("cuda"), labels.to("cuda")

				pred = self.model(data)
				_, pred = torch.max(pred, 1)

				total += labels.size(0)
				correct += (pred == labels).sum().item()

		return correct/total
