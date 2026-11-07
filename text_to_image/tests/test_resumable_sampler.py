"""Data-order resumption (train.py --resume-data-position). CPU only."""

import unittest

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data_utils.resumable_sampler import ResumableDistributedSampler, data_resume_position


class _Index(torch.utils.data.Dataset):
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return i


def _stream(n, world, global_batch, steps, consumed=0, seed=0):
    """Global sample ids per step, as train.py's per-rank loaders would yield them."""
    local = global_batch // world
    loaders, samplers = [], []
    for rank in range(world):
        s = ResumableDistributedSampler(_Index(n), num_replicas=world, rank=rank,
                                        shuffle=True, seed=seed, drop_last=True)
        samplers.append(s)
        loaders.append(DataLoader(_Index(n), batch_size=local, sampler=s, drop_last=True))
    epoch, skip = data_resume_position(consumed, len(loaders[0]), global_batch, local)
    out, iters = [], None
    while len(out) < steps:
        for s in samplers:
            s.set_epoch(epoch)
            if skip:
                s.skip_next_epoch(skip)
        skip = 0
        iters = [iter(dl) for dl in loaders]
        for batches in zip(*iters):
            out.append(sorted(int(i) for b in batches for i in b))
            if len(out) == steps:
                break
        epoch += 1
    return out


class ResumeDataPositionTests(unittest.TestCase):
    N, GB = 10_000, 64

    def test_no_skip_matches_distributed_sampler(self):
        a = ResumableDistributedSampler(_Index(1000), num_replicas=4, rank=1, shuffle=True, seed=3)
        b = DistributedSampler(_Index(1000), num_replicas=4, rank=1, shuffle=True, seed=3)
        for e in (0, 1):
            a.set_epoch(e), b.set_epoch(e)
            self.assertEqual(list(a), list(b))
        self.assertEqual(len(a), len(b))

    def test_resume_continues_stream_even_with_a_different_world_size(self):
        full = _stream(self.N, world=8, global_batch=self.GB, steps=200)  # one uninterrupted run
        k = 70
        resumed = _stream(self.N, world=4, global_batch=self.GB, steps=200 - k, consumed=k * self.GB)
        self.assertEqual(resumed, full[k:])

    def test_resume_across_epoch_boundary(self):
        steps_per_epoch = (self.N // 8) // (self.GB // 8)
        full = _stream(self.N, world=8, global_batch=self.GB, steps=3 * steps_per_epoch)
        k = steps_per_epoch + 11
        resumed = _stream(self.N, world=8, global_batch=self.GB,
                          steps=len(full) - k, consumed=k * self.GB)
        self.assertEqual(resumed, full[k:])

    def test_old_restart_behavior_replays_the_start(self):
        full = _stream(self.N, world=8, global_batch=self.GB, steps=100)
        restarted = _stream(self.N, world=8, global_batch=self.GB, steps=30)  # epoch-0 restart
        self.assertEqual(restarted, full[:30])

    def test_position_arithmetic(self):
        self.assertEqual(data_resume_position(0, 100, 512, 64), (0, 0))
        self.assertEqual(data_resume_position(150 * 512, 100, 512, 64), (1, 50 * 64))
        self.assertEqual(data_resume_position(150 * 512 + 100, 100, 512, 32), (1, 50 * 32))


if __name__ == "__main__":
    unittest.main()
