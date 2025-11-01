"""Data-loading for the MNIST trainer."""



from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader

import xyz.pytorch.sandbox.mnist.data as mnist_data
from xyz.pytorch.sandbox.mnist.train.config import (
    Config,
    DDPConfig,
    FSDPConfig,
)


def create_data_loaders(rank: int, config: Config) -> tuple[DataLoader, DataLoader]:
    """Load MNIST data and return training and test data loaders."""
    mnist_train, mnist_test = mnist_data.load_mnist(config)

    assert config.training_data_fraction == 1.0, "Fractional training data not implemented"

    match config.parallel:
        case None:
            train_sampler, test_sampler = None, None
        case DDPConfig() | FSDPConfig():
            train_sampler = DistributedSampler(
                mnist_train,
                num_replicas=config.parallel.world_size,
                rank=rank,
                seed=config.seed,
                shuffle=config.shuffle,
            )
            test_sampler = DistributedSampler(
                mnist_test,
                num_replicas=config.parallel.world_size,
                rank=rank,
                seed=config.seed,
            )
        case _:
            raise NotImplementedError(f"Parallelism kind {config.parallel} not implemented")

    train_loader = StatefulDataLoader(
        mnist_train,
        sampler=train_sampler,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
    )
    test_loader = StatefulDataLoader(mnist_test, sampler=test_sampler, batch_size=config.batch_size)
    return train_loader, test_loader
