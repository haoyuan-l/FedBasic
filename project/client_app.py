"""Flower ClientApp for federated learning"""

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from task import get_data, train, test
from models import get_model
from typing import Dict, List, Tuple


class FlowerClient(NumPyClient):
    """Basic Flower client for federated learning"""

    def __init__(self, trainloader, testloader, device):
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device
        self.model = get_model().to(device)

    def get_parameters(self, config: Dict[str, str]) -> List:
        """Return current model parameters as numpy arrays"""
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters: List) -> None:
        """Set model parameters from numpy arrays"""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: List, config: Dict[str, str]) -> Tuple[List, int, Dict]:
        """Train the model on local data"""
        # Set model parameters
        self.set_parameters(parameters)

        # Get training config
        local_epochs = int(config.get("local-epochs", 5))
        learning_rate = float(config.get("learning-rate", 0.01))

        # Train the model
        train_loss, train_accuracy = train(
            model=self.model,
            trainloader=self.trainloader,
            epochs=local_epochs,
            learning_rate=learning_rate,
            device=self.device,
        )

        # Return updated parameters and metrics
        updated_parameters = self.get_parameters({})
        metrics = {
            "train_loss": float(train_loss),
            "train_accuracy": float(train_accuracy),
        }

        return updated_parameters, len(self.trainloader.dataset), metrics

    def evaluate(self, parameters: List, config: Dict[str, str]) -> Tuple[float, int, Dict]:
        """Evaluate the model on local test data"""
        # Set model parameters
        self.set_parameters(parameters)

        # Evaluate the model
        test_loss, test_accuracy = test(
            model=self.model,
            testloader=self.testloader,
            device=self.device,
        )

        metrics = {
            "test_accuracy": float(test_accuracy),
        }

        return float(test_loss), len(self.testloader.dataset), metrics


def client_fn(context: Context):
    """Create a Flower client with its data partition"""
    # Get partition configuration
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    # Load data partition
    trainloader, testloader = get_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        batch_size=32,
    )

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create and return client
    return FlowerClient(trainloader, testloader, device).to_client()


# Create ClientApp
app = ClientApp(client_fn=client_fn)
