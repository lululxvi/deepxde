import numpy as np


class BatchSampler:
    """Samples a mini-batch of indices.

    The indices are repeated indefinitely. Has the same effect as:

    .. code-block:: python

        indices = tf.data.Dataset.range(num_samples)
        indices = indices.repeat().shuffle(num_samples).batch(batch_size)
        iterator = iter(indices)
        batch_indices = iterator.get_next()

    However, ``tf.data.Dataset.__iter__()`` is only supported inside of ``tf.function`` or when eager execution is
    enabled. ``tf.data.Dataset.make_one_shot_iterator()`` supports graph mode, but is too slow.

    This class is not implemented as a Python Iterator, so that it can support dynamic batch size.

    Args:
        num_samples (int): The number of samples.
        shuffle (bool): Set to ``True`` to have the indices reshuffled at every epoch.
    """

    def __init__(self, num_samples, shuffle=True):
        self.num_samples = num_samples
        self.shuffle = shuffle

        self._indices = np.arange(self.num_samples)
        self._epochs_completed = 0
        self._index_in_epoch = 0

        # Shuffle for the first epoch
        if shuffle:
            np.random.shuffle(self._indices)

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def get_next(self, batch_size):
        """Returns the indices of the next batch.

        Args:
            batch_size (int): The number of elements to combine in a single batch.
        """
        if batch_size > self.num_samples:
            raise ValueError(
                "batch_size={} is larger than num_samples={}.".format(
                    batch_size, self.num_samples
                )
            )

        start = self._index_in_epoch
        if start + batch_size <= self.num_samples:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._indices[start:end]
        else:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_samples = self.num_samples - start
            indices_rest_part = np.copy(
                self._indices[start : self.num_samples]
            )  # self._indices will be shuffled below.
            # Shuffle the indices
            if self.shuffle:
                np.random.shuffle(self._indices)
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_samples
            end = self._index_in_epoch
            indices_new_part = self._indices[start:end]
            return np.hstack((indices_rest_part, indices_new_part))
