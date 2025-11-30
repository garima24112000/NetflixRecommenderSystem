"""
A small `torch.utils.data.Dataset` implementation that reads from per-column NumPy memmaps
created by `convert_parquet_to_memmap.py`.

Usage:
    ds = MemmapDataset('/path/to/memmap_dir', columns=('userId','movieId','rating'))
    x = ds[10]  # returns dict with numpy scalars

For PyTorch training, wrap with a DataLoader and convert numpy -> torch tensors in collate.
"""
from pathlib import Path
import json
import numpy as np
from torch.utils.data import Dataset


class MemmapDataset(Dataset):
    def __init__(self, memmap_dir, columns=None):
        self.memmap_dir = Path(memmap_dir)
        meta_path = self.memmap_dir / 'meta.json'
        if not meta_path.exists():
            raise FileNotFoundError(f'meta.json not found in {memmap_dir}')
        meta = json.loads(meta_path.read_text())
        self.total_rows = int(meta['total_rows'])
        self.columns = columns if columns is not None else meta['columns']
        self.dtypes = meta.get('dtypes', {})
        # open memmaps in read-only mode
        self._memmaps = {}
        for col in self.columns:
            fname = self.memmap_dir / f'{col}.npy'
            if not fname.exists():
                raise FileNotFoundError(f'Expected memmap file missing: {fname}')
            # Use numpy.load with mmap_mode='r' to get an mmap-backed array
            arr = np.load(str(fname), mmap_mode='r')
            self._memmaps[col] = arr

    def __len__(self):
        return self.total_rows

    def __getitem__(self, idx):
        if idx < 0:
            idx = self.total_rows + idx
        if idx < 0 or idx >= self.total_rows:
            raise IndexError(idx)
        # return a dict of scalars (numpy types) per column
        out = {col: self._memmaps[col][idx] for col in self.columns}
        return out


# helper collate that converts numpy -> torch tensors
def collate_numpy_to_tensors(batch, torch):
    """Convert a list of dicts (from __getitem__) into tensors. Returns (inputs, targets) pattern.
    Example: columns = [userId, movieId, rating] -> inputs=(user_tensor, movie_tensor), targets=rating_tensor
    The user of this helper must provide which columns are inputs vs target.
    """
    if not batch:
        return {}
    keys = batch[0].keys()
    stacked = {}
    for k in keys:
        arr = np.stack([item[k] for item in batch])
        stacked[k] = torch.from_numpy(arr)
    return stacked


if __name__ == '__main__':
    # quick local sanity check (not a unit test)
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('memmap_dir')
    args = p.parse_args()
    ds = MemmapDataset(args.memmap_dir)
    print('Dataset size:', len(ds))
    print('Sample[0]:', ds[0])
