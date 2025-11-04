"""Alternative way to run FL simulation directly with Python"""

import torch
from flwr.simulation import run_simulation
from client_app import app as client_app
from server_app import app as server_app
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)


def main():
    """Run Federated Learning simulation with SimpleCNN on CIFAR-10"""

    # Simulation parameters
    NUM_CLIENTS = 10
    NUM_ROUNDS = 10

    # Backend configuration
    backend_config = {
        "client_resources": {
            "num_cpus": 2,
            "num_gpus": 0.0,  # Set to 0.25 if using GPU
        }
    }

    logger.info("Starting FL simulation: SimpleCNN on CIFAR-10")
    logger.info(f"Clients: {NUM_CLIENTS}, Rounds: {NUM_ROUNDS}")

    # Run simulation
    history = run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=NUM_CLIENTS,
        backend_config=backend_config,
    )

    logger.info("Simulation completed")

    # Display results
    if hasattr(history, 'losses_centralized') and history.losses_centralized:
        logger.info("\nCentralized Evaluation:")
        for round_num, (loss, metrics) in enumerate(history.losses_centralized, 1):
            accuracy = metrics.get('accuracy', 0.0)
            logger.info(f"  Round {round_num}: Loss={loss:.4f}, Accuracy={accuracy:.2f}%")

    if hasattr(history, 'losses_distributed') and history.losses_distributed:
        logger.info("\nDistributed Evaluation:")
        for round_num, (loss, metrics) in enumerate(history.losses_distributed, 1):
            accuracy = metrics.get('accuracy', 0.0)
            logger.info(f"  Round {round_num}: Loss={loss:.4f}, Accuracy={accuracy:.2f}%")

    return history


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    history = main()
