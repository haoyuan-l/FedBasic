import flwr as fl
import torch
import collections
import torchmetrics
import torch.optim as optim
from data import *

class Client(fl.client.NumPyClient):

	def __init__(self, cid, dataset, num_clients, model_loader, encoder, data_split, skewness_alpha, embed_input=False, device='cuda', fl_method="fedavg", seed=42):
		self.fl_method = fl_method
		self.cid = cid
		self.data, self.num_classes, self.num_samples, self.ratio = get_data(dataset_name=dataset, id=cid, num_clients=num_clients, seed=seed,
																	embed_input=embed_input, encoder=encoder, split=data_split, alpha=skewness_alpha)
		self.embed_input = embed_input
		# Determine input shape based on embed_input flag
		if self.embed_input:
			try:
				first_batch = next(iter(self.data))
				first_embedding = first_batch[0]
				if isinstance(first_embedding, torch.Tensor):
					emb_dim = first_embedding.shape[-1]
					self.input_shape = (emb_dim,)
				else:
					raise ValueError("Expected embedding to be a torch.Tensor")
			except StopIteration:
				raise ValueError("DataLoader is empty. Cannot determine embedding dimension.")
		else:
			self.input_shape = self.get_dataset_config(dataset)
		self.model_loader = model_loader
		self.embed_input = embed_input
		self.device = device

	def set_parameters(self, parameters, config):
		if not hasattr(self, 'model'):
			self.model = self.model_loader(input_shape=self.input_shape, num_classes=self.num_classes).to(self.device)
		params_dict = zip(self.model.state_dict().keys(), parameters)
		state_dict = collections.OrderedDict({k: torch.tensor(v) for k, v in params_dict})
		self.model.load_state_dict(state_dict, strict=True)

	def get_parameters(self, config={}):
		return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

	def fit(self, parameters, config):
		self.set_parameters(parameters, config)
		# SGD optimizer
		lr = config.get('lr', 1e-3) 
		optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
		# Adam optimizer
		#optimizer = torch.optim.Adam(self.model.parameters(), lr=config['lr'])
		if self.fl_method.lower() == "fedavg":
			h = __class__.train(ds=self.data, model=self.model, epochs=config['epochs'], optimizer=optimizer, num_classes=self.num_classes)
			return self.get_parameters(), self.num_samples, h

		elif self.fl_method.lower() == "fedprox":
			global_params = [p.clone().detach() for p in self.model.parameters()]
			proximal_mu=config["proximal_mu"]
			h = __class__.fedprox_train(ds=self.data, model=self.model, epochs=config['epochs'], optimizer=optimizer, num_classes=self.num_classes, global_params=global_params, proximal_mu=proximal_mu)
			return self.get_parameters(), self.num_samples, h

		elif self.fl_method.lower() == "fednova":
			h = __class__.fednova_train(ds=self.data, model=self.model, epochs=config['epochs'], optimizer=optimizer, num_classes=self.num_classes)
			local_tau = h['local_normalizing_vec'] * self.ratio
			return self.get_parameters({}), self.num_samples, {'loss': h['loss'], 'accuracy': h['accuracy'], "ratio": self.ratio, "tau": local_tau, "local_norm": h['local_normalizing_vec']}

	def evaluate(self, parameters, config):
		raise NotImplementedError('Client-side evaluation is not implemented!')
	
	def get_parameters(self, config={}):
		return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
	
    # Get the input shape based on the dataset
	def get_dataset_config(self, dataset):
		if dataset.lower() in ["cifar10", "svhn", "cifar100"]:
			input_shape=(3, 32, 32)
		elif dataset.lower() in ["pathmnist", "dermamnist"]:
			input_shape=(3, 28, 28)
		elif dataset.lower() == "tiny-imagenet":
			input_shape = (3, 64, 64)
		else:
			raise NotImplementedError(f"Dataset '{dataset}' is not supported.")
		return input_shape

	@staticmethod
	def train(ds, model, epochs, optimizer, num_classes, metrics=None, loss=torch.nn.CrossEntropyLoss(), verbose=False):
		device = next(model.parameters()).device
		if metrics is None:
			metrics = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes, average='micro').to(device)
		loss_scores = []
		model.train()
		for epoch in range(epochs):
			train_loss = 0.0
			for _, (x, y) in enumerate(ds):
				x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True).long().squeeze()
				optimizer.zero_grad()
				preds = model(x)
				_loss = loss(preds, y)
				_loss.backward()
				optimizer.step()
				train_loss += _loss.item()
				metrics(preds.max(1)[-1], y)
			train_loss /= len(ds)
			loss_scores.append(train_loss)
			acc = metrics.compute()
			if verbose:
				print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - Accuracy: {100. * acc:.2f}%")
		return {'loss': loss_scores, 'accuracy': acc}
	
	@staticmethod
	def fednova_train(ds, model, epochs, optimizer, num_classes, metrics=None, loss=torch.nn.CrossEntropyLoss(), verbose=False):
		# Track the local updates
		device = next(model.parameters()).device
		local_normalizing_vec = 0
		local_counter = 0
		local_steps = 0
		if metrics is None:
			metrics = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes, average='micro').to(device)
		loss_scores = []
		model.train()
		for epoch in range(epochs):
			train_loss = 0.0
			for _, (x, y) in enumerate(ds):
				x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True).long().squeeze()
				optimizer.zero_grad()
				preds = model(x)
				_loss = loss(preds, y)
				_loss.backward()
				optimizer.step()

				# Update local stats
				local_counter = local_counter * 0.9 + 1 # SGD momentum=0.9
				local_normalizing_vec += local_counter

				train_loss += _loss.item()
				metrics(preds.max(1)[-1], y)
			train_loss /= len(ds)
			loss_scores.append(train_loss)
			acc = metrics.compute()
			if verbose:
				print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - Accuracy: {100. * acc:.2f}%")

		return {'loss': loss_scores, 'accuracy': acc, 'local_normalizing_vec': local_normalizing_vec}


	@staticmethod
	def fedprox_train(ds, model, epochs, optimizer, num_classes, global_params, proximal_mu, metrics=None, loss=torch.nn.CrossEntropyLoss(), verbose=False):
		device = next(model.parameters()).device
		if metrics is None:
			metrics = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes, average='micro').to(device)
		loss_scores = []
		model.train()
		for epoch in range(epochs):
			train_loss = 0.0
			for _, (x, y) in enumerate(ds):
				x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True).long().squeeze()
				optimizer.zero_grad()
				preds = model(x)
				_loss = loss(preds, y)

				# Proximal term calculation
				proximal_term = 0.0
				for param, global_param in zip(model.parameters(), global_params):
					proximal_term += (param - global_param).norm(2)

				# Total loss with proximal term
				total_loss = _loss + (proximal_mu / 2) * proximal_term
				total_loss.backward()
				optimizer.step()
				#scheduler.step()
				train_loss += total_loss.item()
				metrics(preds.max(1)[-1], y)
			train_loss /= len(ds)
			loss_scores.append(train_loss)
			acc = metrics.compute()
			if verbose:
				print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - Accuracy: {100. * acc:.2f}%")

		return {'loss': loss_scores, 'accuracy': acc}