import time
from pathlib import Path
import random
import torch
from torch.utils.data import IterableDataset, get_worker_info, DataLoader
import multiprocessing as mp


class BufferedBatchedDataset(IterableDataset):
    def __init__(
        self,
        data_dir,
        buffer_size=100_000,
        seed=42,
        file_pattern="*.pt",
        shuffle_files=True,
        max_files=50,
        batch_size=64,
    ):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.files = sorted(self.data_dir.glob(file_pattern))
        self.files = self.files[:max_files]
        self.batch_size = batch_size

        if len(self.files) == 0:
            raise ValueError(f"No files matching {file_pattern} found in {data_dir}")

        self.buffer_size = buffer_size
        self.seed = seed
        self.shuffle_files = shuffle_files

    def _read_file(self, path):
        x = torch.load(path)

        if not torch.is_tensor(x):
            raise TypeError(f"Expected tensor in {path}, got {type(x)}")

        if x.ndim != 2:
            raise ValueError(f"Expected 2D tensor in {path}, got shape {tuple(x.shape)}")

        for i in range(x.shape[0]):
            yield x[i]

    def __iter__(self):
        worker = get_worker_info()

        files = self.files

        if worker is not None:
            files = files[worker.id :: worker.num_workers]
            rng = random.Random(self.seed + worker.id)
        else:
            rng = random.Random(self.seed)

        files = list(files)

        if self.shuffle_files:
            rng.shuffle(files)

        buffer = []
        batch_buffer = []

        for path in files:
            for sample in self._read_file(path):
                if len(buffer) < self.buffer_size:
                    buffer.append(sample)
                else:
                    j = rng.randrange(self.buffer_size)
                    chosen = buffer[j]
                    buffer[j] = sample
                    batch_buffer.append(chosen)
                    if len(batch_buffer) == self.batch_size:
                        yield torch.stack(batch_buffer)
                        batch_buffer.clear()
        rng.shuffle(buffer)
        if len(buffer) >= self.batch_size:
            for i in range(0, len(buffer), self.batch_size):
                yield torch.stack(buffer[i: i + self.batch_size])
        buffer.clear()

class BufferedDataset(IterableDataset):
    def __init__(
        self,
        data_dir,
        buffer_size=100_000,
        seed=42,
        file_pattern="*.pt",
        shuffle_files=True,
        max_files=50,
    ):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.files = sorted(self.data_dir.glob(file_pattern))
        self.files = self.files[:max_files]

        if len(self.files) == 0:
            raise ValueError(f"No files matching {file_pattern} found in {data_dir}")

        self.buffer_size = buffer_size
        self.seed = seed
        self.shuffle_files = shuffle_files

    def _read_file(self, path):
        x = torch.load(path, map_location="cpu")

        if not torch.is_tensor(x):
            raise TypeError(f"Expected tensor in {path}, got {type(x)}")

        if x.ndim != 2:
            raise ValueError(f"Expected 2D tensor in {path}, got shape {tuple(x.shape)}")

        for i in range(x.shape[0]):
            yield x[i]

    def __iter__(self):
        worker = get_worker_info()

        files = self.files

        if worker is not None:
            files = files[worker.id :: worker.num_workers]
            rng = random.Random(self.seed + worker.id)
        else:
            rng = random.Random(self.seed)

        files = list(files)

        if self.shuffle_files:
            rng.shuffle(files)

        buffer = []

        for path in files:
            for sample in self._read_file(path):
                if len(buffer) < self.buffer_size:
                    buffer.append(sample)
                else:
                    j = rng.randrange(len(buffer))
                    yield buffer[j]
                    buffer[j] = sample

        rng.shuffle(buffer)
        yield from buffer


