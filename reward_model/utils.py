import numpy as np
import json

class RunningStat(object):
        def __init__(self, shape=()):
            self._n = 0
            self._M = np.zeros(shape)
            self._S = np.zeros(shape)

        def push(self, x):
            x = np.asarray(x)
            assert x.shape == self._M.shape
            self._n += 1
            if self._n == 1:
                self._M[...] = x
            else:
                oldM = self._M.copy()
                self._M[...] = oldM + (x - oldM)/self._n
                self._S[...] = self._S + (x - oldM)*(x - self._M)

        @property
        def n(self):
            return self._n

        @property
        def mean(self):
            return self._M

        @property
        def var(self):
            if self._n >= 2:
                return self._S/(self._n - 1)
            else:
                return np.square(self._M)

        @property
        def std(self):
            return np.sqrt(self.var)

        @property
        def shape(self):

            return self._M.shape

class LimitedRunningStat(object):
    def __init__(self, len=1000):
        self.values = np.array(np.zeros(len))
        self.n_values = 0
        self.i = 0
        self.len = len

    def push(self, x):
        self.values[self.i] = x
        self.i = (self.i + 1) % len(self.values)
        if self.n_values < len(self.values):
            self.n_values += 1

    @property
    def n(self):
        return self.n_values

    @property
    def mean(self):
        return np.mean(self.values[:self.n_values])

    @property
    def var(self):
        return np.var(self.values[:self.n_values])

    @property
    def std(self):
        return np.std(self.values[:self.n_values])

class DynamicRunningStat(object):

    def __init__(self):
        self.current_rewards = list()
        self.next_rewards = list()

    def push(self, x):
        self.next_rewards.append(x)

    def reset(self):
        self.current_rewards = self.next_rewards
        self.next_rewards = list()

    @property
    def n(self):
        return len(self.current_rewards)

    @property
    def mean(self):
        return np.mean(np.asarray(self.current_rewards))

    @property
    def std(self):
        return np.std(np.asarray(self.current_rewards))


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
