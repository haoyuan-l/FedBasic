"""Flower ServerApp for federated learning"""

import torch
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.common import Context, ndarrays_to_parameters
from flwr.server.strategy import FedAvg
from task import get_data, test
from models import get_model


def gen_evaluate_fn(testloader, device):
    """Generate centralized evaluation function"""

    def evaluate(server_round, parameters_ndarrays, config):
        """Evaluate global model on centralized test set"""
        # Load model and parameters
        model = get_model()
        params_dict = zip(model.state_dict().keys(), parameters_ndarrays)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)

        # Evaluate
        if isinstance(device, str):
            eval_device = torch.device(device)
        else:
            eval_device = device

        loss, accuracy = test(model, testloader, eval_device)

        metrics = {
            "accuracy": float(accuracy),
            "server_round": float(server_round),
        }

        return float(loss), metrics

    return evaluate


def on_fit_config(server_round: int):
    """Configure training for each round"""
    config = {
        "server_round": float(server_round),
        "local-epochs": 5.0,
        "learning-rate": 0.01,
    }
    return config


def server_fn(context: Context):
    """Create server components for federated learning"""
    # Initialize model
    model = get_model()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Get initial parameters
    initial_parameters = ndarrays_to_parameters(
        [val.cpu().numpy() for val in model.state_dict().values()]
    )

    # Get number of rounds and clients from config
    num_rounds = context.run_config.get("num-server-rounds", 10)
    num_clients = context.run_config.get("num-supernodes", 10)

    # Create centralized test set (using partition 0 for simplicity)
    _, testloader = get_data(
        partition_id=0,
        num_partitions=num_clients,
        batch_size=64,
    )

    # Create evaluation function
    evaluate_fn = gen_evaluate_fn(testloader, device)

    # Configure FedAvg strategy
    strategy = FedAvg(
        fraction_fit=1.0,  # Use all available clients for training
        fraction_evaluate=1.0,  # Use 100% of clients for evaluation
        min_fit_clients=min(num_clients, 2),
        min_evaluate_clients=num_clients,  # Require all clients for evaluation
        min_available_clients=min(num_clients, 2),
        evaluate_fn=evaluate_fn,
        on_fit_config_fn=on_fit_config,
        initial_parameters=initial_parameters,
    )

    # Create server config
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
