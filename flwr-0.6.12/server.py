import logging
import flwr as fl
import torch
from data import *
import collections
import torchmetrics
from strategy import *
from utils import get_learning_rate

class Server(fl.server.Server):

	def __init__(self, dataset, model_loader, encoder, data_split, skewness_alpha, return_eval_ds, num_rounds, num_clients=10, embed_input=False, participation=1.0, 
				 init_model=None, log_level=logging.INFO, initial_lr=1e-3, decay_factor=0.1, num_decays=3, fl_method="fedavg", seed=42, local_epochs=5):
		
		self.fl_method = fl_method
		self.num_rounds = num_rounds
		self.data, self.num_classes, self.num_samples = get_data(dataset_name=dataset, num_clients=num_clients, embed_input=embed_input, encoder=encoder, 
														   split=data_split, seed=seed, alpha=skewness_alpha, return_eval_ds=return_eval_ds, )
		
		self.embed_input = embed_input
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
		self.init_model = init_model
		self.initial_lr = initial_lr
		self.decay_factor = decay_factor
		self.num_decays = num_decays
		self.clients_config = {"epochs":local_epochs, "lr":initial_lr}
		self.num_clients = num_clients
		self.participation = participation
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self._client_manager = fl.server.client_manager.SimpleClientManager()
		self.max_workers = None
		self.set_strategy(self)
		self.embed_input = embed_input
		logging.getLogger("flower").setLevel(log_level)

	def set_max_workers(self, *args, **kwargs):
		return super(Server, self).set_max_workers(*args, **kwargs)

	def set_strategy(self, *_):
		if self.fl_method.lower() == "fedavg":
			self.strategy = fl.server.strategy.FedAvg(
				min_available_clients=self.num_clients, fraction_fit=self.participation,
				min_fit_clients=int(self.participation*self.num_clients), fraction_evaluate=0.0,
				min_evaluate_clients=0, evaluate_fn=self.get_evaluation_fn(),
				on_fit_config_fn=self.get_client_config_fn(), initial_parameters=self.get_initial_parameters(),)
		elif self.fl_method.lower() == "fedprox":
			self.strategy = fl.server.strategy.FedProx(
			min_available_clients=self.num_clients, fraction_fit=self.participation,
			min_fit_clients=int(self.participation*self.num_clients), fraction_evaluate=0.0,
			min_evaluate_clients=0, evaluate_fn=self.get_evaluation_fn(),
			on_fit_config_fn=self.get_client_config_fn(), initial_parameters=self.get_initial_parameters(),
			proximal_mu=0.1,)
		elif self.fl_method.lower() == "fednova":
			self.strategy = CustomFedNova(
				min_available_clients=self.num_clients, fraction_fit=self.participation,
				min_fit_clients=int(self.participation*self.num_clients), fraction_evaluate=0.0,
				min_evaluate_clients=0, evaluate_fn=self.get_evaluation_fn(),
				on_fit_config_fn=self.get_client_config_fn(), initial_parameters=self.get_initial_parameters(),)

	def client_manager(self, *args, **kwargs):
		return super(Server, self).client_manager(*args, **kwargs)

	def get_parameters(self, config={}):
		return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
	
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

	def set_parameters(self, parameters, config):
		if not hasattr(self, 'model'):
			self.model = self.model_loader(input_shape=self.input_shape, num_classes=self.num_classes).to(self.device)
		params_dict = zip(self.model.state_dict().keys(), parameters)
		state_dict = collections.OrderedDict({k: torch.tensor(v) for k, v in params_dict})
		self.model.load_state_dict(state_dict, strict=True)

	def get_initial_parameters(self, *_):
		""" Get initial random model weights """
		if self.init_model is not None:
			self.init_weights = torch.load(self.init_model, map_location=self.device).state_dict()
		else:
			self.init_weights = [val.cpu().numpy() for _, val in self.model_loader(input_shape=self.input_shape, num_classes=self.num_classes).state_dict().items()]
		return fl.common.ndarrays_to_parameters(self.init_weights)

	def get_evaluation_fn(self):
		def evaluation_fn(rnd, parameters, config):
			self.set_parameters(parameters, config)
			metrics = __class__.evaluate(model=self.model, ds=self.data, num_classes=self.num_classes)
			return metrics[0], {"accuracy":metrics[1]}
		return evaluation_fn

	def get_client_config_fn(self):
		"""Define fit config function with dynamic learning rate based on round."""
		def config_fn(rnd):
            # Calculate the learning rate based on the current round
			current_lr = get_learning_rate(
                initial_lr=self.initial_lr,
				current_round=rnd,
                total_rounds=self.num_rounds,
                decay_factor=self.decay_factor,
                num_decays=self.num_decays
            )
            # Update the clients' configuration
			client_config = {
				"epochs": 5,
                "lr": current_lr,
                "round": rnd
            }
			logging.info(f"Round {rnd}: Setting client learning rate to {current_lr}")
			return client_config
		return config_fn

	@staticmethod
	def evaluate(ds, model, num_classes, metrics=None, loss=torch.nn.CrossEntropyLoss(), verbose=False):
		device = next(model.parameters()).device
		if metrics is None:
			metrics = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes, average='micro').to(device)
		model.eval()
		_loss = 0.0
		with torch.no_grad():
			for _, (x, y) in enumerate(ds):
				x, y = x.to(device), y.to(device).long().squeeze()
				preds = model(x)
				_loss += loss(preds, y).item()
				metrics(preds.max(1)[-1], y)
		_loss /= len(ds)
		acc = metrics.compute()
		if verbose:
			print(f"Loss: {_loss:.4f} - Accuracy: {100. * acc:.2f}%")
		return (_loss, acc)