"""One active/one pending chunk. Callers serialize with their session lock."""
from collections import deque
from dataclasses import dataclass

import numpy as np


class ActionContractError(ValueError):
    pass


@dataclass(frozen=True)
class ActionContract:
    width: int
    horizon: int = None

    def validate(self, actions, minimum=1, finite_rows=None):
        try:
            value = np.asarray(actions, dtype=np.float64)
        except (ValueError, TypeError) as exc:
            raise ActionContractError("invalid policy actions") from exc
        if (value.ndim != 2 or value.shape[1] != self.width or value.shape[0] < minimum
                or (self.horizon is not None and value.shape[0] != self.horizon)):
            raise ActionContractError("expected policy actions[%s,%d], got %s" %
                                      (self.horizon or "N", self.width, value.shape))
        if not np.isfinite(value if finite_rows is None else value[:finite_rows]).all():
            raise ActionContractError("policy action chunk contains NaN or infinity")
        return value


@dataclass
class Session:
    generation: int = 0
    enabled: bool = False
    prompt: str = None

    def accepts(self, generation):
        return self.enabled and generation == self.generation

    def advance(self, enabled):
        self.generation += 1
        self.enabled = enabled
        return self.generation


class ChunkQueue:
    def __init__(self, length, prefetch, contract, reset_ids=True, underrun_error=True):
        if length < 1 or (contract.horizon is not None and length > contract.horizon) or not 0 <= prefetch < length:
            raise ValueError("invalid execution chunk/prefetch")
        self.length, self.prefetch, self.contract = length, prefetch, contract
        self.reset_ids, self.underrun_error = reset_ids, underrun_error
        self.session = Session()
        self.next_id = 1
        self.awaiting_initial = False
        self.clear()

    generation = property(lambda self: self.session.generation,
                          lambda self, value: setattr(self.session, "generation", value))
    enabled = property(lambda self: self.session.enabled,
                       lambda self, value: setattr(self.session, "enabled", value))

    def clear(self):
        self.active, self.pending = deque(), None
        self.index, self.chunk_id = 0, 0
        if self.reset_ids:
            self.next_id = 1

    def start(self):
        generation = self.session.advance(True)
        self.clear()
        self.awaiting_initial = True
        return generation

    def stop(self):
        self.session.advance(False)
        self.awaiting_initial = False
        self.clear()

    def stage(self, actions, generation, request=None):
        if not self.session.accepts(generation):
            return False
        if request is not None:
            request.check()
        data = self.contract.validate(actions, minimum=self.length,
                                      finite_rows=None if self.contract.horizon else self.length)
        if self.pending is not None:
            raise ValueError("pending_chunk_already_exists")
        pending = (self.next_id, data[:self.length].copy())
        if request is not None:
            request.check()  # Validation/copy time is inside the acceptance deadline.
        self.pending = pending
        self.next_id += 1
        return True

    def select(self):
        if not self.enabled:
            return None
        if not self.active:
            if self.pending is None:
                if self.awaiting_initial or not self.underrun_error:
                    return None
                raise ValueError("chunk_underrun")
            self.chunk_id, data = self.pending
            self.pending = None
            self.active = deque(data)
            self.index, self.awaiting_initial = 0, False
        index = self.index
        self.index += 1
        return self.chunk_id, index, self.active.popleft(), index == self.prefetch
