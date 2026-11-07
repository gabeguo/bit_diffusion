"""Resume the training-data order where a checkpointed run stopped.

``DistributedSampler`` draws one global permutation per epoch (seeded by
``seed + epoch``) and gives rank ``r`` the strided slice ``perm[r::world]``. A
step with per-rank batch ``b`` therefore consumes global permutation positions
``[s*B, (s+1)*B)`` with ``B = world * b``, whatever the world size. Resuming
after ``n`` consumed samples is just: start at epoch ``n // samples_per_epoch``
and drop the first ``(n % samples_per_epoch) / world`` indices of each rank's
slice. Only indices are skipped, never data, so restoring costs about as much
as an ordinary epoch start (one permutation of the dataset).
"""

from __future__ import annotations

import itertools

from torch.utils.data.distributed import DistributedSampler


class ResumableDistributedSampler(DistributedSampler):
    """``DistributedSampler`` whose next epoch can start part-way through.

    With no pending skip it yields exactly what ``DistributedSampler`` yields.
    ``__len__`` stays the full per-epoch length so step accounting is unchanged.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._skip = 0

    def skip_next_epoch(self, num_indices: int) -> None:
        """Drop the first ``num_indices`` of this rank's indices on the next ``__iter__``."""
        if num_indices < 0:
            raise ValueError("num_indices must be nonnegative")
        self._skip = int(num_indices)

    def __iter__(self):
        indices = super().__iter__()
        skip, self._skip = self._skip, 0
        return itertools.islice(indices, skip, None) if skip else indices


def data_resume_position(samples_consumed: int, steps_per_epoch: int,
                         global_batch: int, local_batch: int) -> tuple[int, int]:
    """(epoch, per-rank indices to skip) after ``samples_consumed`` global samples.

    ``steps_per_epoch`` is the per-rank batch count of one epoch (the length of
    the drop_last training loader). Skips are whole steps, so batches stay
    aligned; a partially consumed step (only possible if the global batch
    changed) is repeated rather than skipped.
    """
    if steps_per_epoch <= 0 or global_batch <= 0 or local_batch <= 0:
        raise ValueError("steps_per_epoch, global_batch and local_batch must be positive")
    epoch, within = divmod(int(samples_consumed), steps_per_epoch * global_batch)
    return epoch, (within // global_batch) * local_batch
