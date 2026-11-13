"""Uniform replay that shares overlapping uint8 frames between observations.

The transition budget has the same meaning as a deque's ``maxlen``. Frames are
owned, immutable copies; sampling reconstructs independent stacked observations.
Only equal frames are shared, so reset and truncation boundaries need no special
handling. This is a storage change, not prioritized experience replay.
"""

import math
import operator
import random

import numpy as np


class FrameReplayBuffer:
    """A transition ring with shared frame arrays and uniform sampling.

    Observations must be nonempty uint8 arrays with their stack on axis zero.
    Their shape is fixed by the first transition. ``sample`` draws without
    replacement using Python's global random generator, preserved in the full
    training checkpoint. Instances can be pickled by ``torch.save`` directly.
    """

    def __init__(self, capacity):
        if isinstance(capacity, (bool, np.bool_)):
            raise ValueError("replay capacity must be a positive integer")
        try:
            capacity = operator.index(capacity)
        except TypeError as error:
            raise ValueError("replay capacity must be a positive integer") from error
        if capacity <= 0:
            raise ValueError("replay capacity must be a positive integer")
        self.capacity = capacity
        self._records = [None] * capacity
        self._next = 0
        self._size = 0
        self._shape = None
        self._last_next_frames = None

    @property
    def maxlen(self):
        return self.capacity

    def __len__(self):
        return self._size

    @staticmethod
    def _same_stack(observation, frames):
        return (frames is not None and len(observation) == len(frames)
                and all(np.array_equal(value, frame)
                        for value, frame in zip(observation, frames)))

    @staticmethod
    def _copy_or_share(observation, candidates=()):
        """Own new frames, reusing only candidates whose values are equal."""
        frames = []
        for value in observation:
            match = next((frame for frame in (*candidates, *frames)
                          if np.array_equal(value, frame)), None)
            if match is None:
                match = np.array(value, dtype=np.uint8, copy=True)
                match.flags.writeable = False
            frames.append(match)
        return tuple(frames)

    def _validate(self, transition):
        try:
            state, action, reward, next_state, terminal = transition
        except (TypeError, ValueError) as error:
            raise ValueError("a replay transition must have five fields") from error
        for observation in (state, next_state):
            if (not isinstance(observation, np.ndarray)
                    or observation.dtype != np.uint8
                    or observation.ndim < 1 or observation.size == 0):
                raise ValueError("replay observations must be nonempty uint8 arrays")
        if state.shape != next_state.shape:
            raise ValueError("state and next_state must have the same shape")
        if self._shape is not None and state.shape != self._shape:
            raise ValueError("observation shape differs from this replay buffer")
        if isinstance(action, (bool, np.bool_)):
            raise ValueError("replay action must be a nonnegative integer")
        try:
            action = operator.index(action)
        except TypeError as error:
            raise ValueError("replay action must be a nonnegative integer") from error
        if action < 0:
            raise ValueError("replay action must be a nonnegative integer")
        if isinstance(reward, (bool, np.bool_, str, bytes)):
            raise ValueError("replay reward must be a finite real number")
        try:
            reward = float(reward)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("replay reward must be a finite real number") from error
        if not math.isfinite(reward):
            raise ValueError("replay reward must be a finite real number")
        if not isinstance(terminal, (bool, np.bool_)):
            raise ValueError("replay terminal flag must be boolean")
        return state, action, reward, next_state, bool(terminal)

    def append(self, transition):
        state, action, reward, next_state, terminal = self._validate(transition)
        if self._same_stack(state, self._last_next_frames):
            state_frames = self._last_next_frames
        else:
            state_frames = self._copy_or_share(state)
        # The usual frame-stack step shifts left and appends one new frame.
        # Validate the overlap rather than assuming continuity across resets.
        if (len(state_frames) > 1
                and self._same_stack(next_state[:-1], state_frames[1:])):
            final = self._copy_or_share(next_state[-1:], state_frames)[0]
            next_frames = (*state_frames[1:], final)
        else:
            next_frames = self._copy_or_share(next_state, state_frames)
        record = (state_frames, action, reward, next_frames, terminal)
        # Everything above can fail without altering the buffer or its cache.
        self._records[self._next] = record
        self._next = (self._next + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
        self._shape = state.shape
        self._last_next_frames = next_frames

    def _record(self, index):
        if isinstance(index, (bool, np.bool_)):
            raise TypeError("replay index must be an integer")
        index = operator.index(index)
        if index < 0:
            index += self._size
        if not 0 <= index < self._size:
            raise IndexError("replay index out of range")
        start = (self._next - self._size) % self.capacity
        return self._records[(start + index) % self.capacity]

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(self._size))]
        state, action, reward, next_state, terminal = self._record(index)
        return np.stack(state), action, reward, np.stack(next_state), terminal

    def __iter__(self):
        for index in range(self._size):
            yield self[index]

    def sample(self, batch_size):
        """Return independent transitions chosen uniformly without replacement."""
        if isinstance(batch_size, (bool, np.bool_)):
            raise ValueError("batch_size must be an integer between zero and replay size")
        try:
            batch_size = operator.index(batch_size)
        except TypeError as error:
            raise ValueError("batch_size must be an integer between zero and replay size") from error
        if not 0 <= batch_size <= self._size:
            raise ValueError("batch_size must be an integer between zero and replay size")
        return [self[index] for index in random.sample(range(self._size), batch_size)]

    @property
    def stored_frame_bytes(self):
        """Count unique retained NumPy frame bytes (excluding Python metadata).

        This diagnostic scans the ring; do not call it on every training step.
        """
        frames = {}
        for record in self._records:
            if record is not None:
                for frame in (*record[0], *record[3]):
                    frames[id(frame)] = frame
        if self._last_next_frames is not None:
            for frame in self._last_next_frames:
                frames[id(frame)] = frame
        return sum(frame.nbytes for frame in frames.values())
