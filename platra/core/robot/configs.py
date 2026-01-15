from typing import Self

import numpy as np

from .ackermann import AckermannConfig, AckermannState, AckermannStateExt


class AckermannConfigForStaticFeedback(AckermannConfig):
    def __init__(self, Lf: float, Ls: float, e: float, r: float) -> None:
        super().__init__(Lf, Ls, e, r)
        self.rts_fn = None
        self.h_fn = None
        self.k_inv_fn = None
        self.g_fn = None

    def rts(self, state: AckermannState) -> np.ndarray:
        assert callable(self.rts_fn)
        return self.rts_fn(state)

    def h(self, state: AckermannState) -> np.ndarray:
        assert callable(self.h_fn)
        return self.h_fn(state)

    def k_inv(self, state: AckermannState) -> np.ndarray:
        assert callable(self.k_inv_fn)
        return self.k_inv_fn(state)

    def g(self, state: AckermannState) -> np.ndarray:
        assert callable(self.g_fn)
        return self.g_fn(state)

    def clip_beta_s(self, beta_s: float) -> tuple[int, float]:
        minimum = 0
        if beta_s <= minimum:
            return 1, minimum
        maximum = -2 * self.alpha3s
        if beta_s >= maximum:
            return 2, maximum
        return 0, beta_s

    @classmethod
    def from_symbolic(
        cls,
        Lf: float,
        Ls: float,
        e: float,
        r: float,
        symbolic_model,
    ) -> Self:
        self = cls(Lf, Ls, e, r)

        num = symbolic_model(self)

        self.rts_fn = num.rts_fn
        self.h_fn = num.h_fn
        self.k_inv_fn = num.k_inv_fn
        self.g_fn = num.g_fn

        return self


class AckermannConfigForDynamicFeedback(AckermannConfig):
    def __init__(self, Lf: float, Ls: float, e: float, r: float) -> None:
        super().__init__(Lf, Ls, e, r)
        self.rts_fn = None
        self.h_fn = None
        self.h_dot_fn = None
        self.h_ddot_fn = None
        self.a_inv_fn = None
        self.b_fn = None

    def rts(self, state: AckermannState) -> np.ndarray:
        assert callable(self.rts_fn)
        return self.rts_fn(state)

    def h(self, state: AckermannState) -> np.ndarray:
        assert callable(self.h_fn)
        return self.h_fn(state)

    def h_dot(self, state: AckermannState) -> np.ndarray:
        assert callable(self.h_dot_fn)
        return self.h_dot_fn(state)

    def h_ddot(self, state: AckermannStateExt) -> np.ndarray:
        assert callable(self.h_ddot_fn)
        return self.h_ddot_fn(state)

    def a_inv(self, state: AckermannState) -> np.ndarray:
        assert callable(self.a_inv_fn)
        return self.a_inv_fn(state)

    def b(self, state: AckermannStateExt) -> np.ndarray:
        assert callable(self.b_fn)
        return self.b_fn(state)

    def clip_beta_s(self, beta_s: float) -> tuple[int, float]:
        minimum = 0
        if beta_s <= minimum:
            return 1, minimum
        maximum = -2 * self.alpha3s
        if beta_s >= maximum:
            return 2, maximum
        return 0, beta_s

    @classmethod
    def from_symbolic(
        cls,
        Lf: float,
        Ls: float,
        e: float,
        r: float,
        symbolic_model,
    ) -> Self:
        self = cls(Lf, Ls, e, r)

        num = symbolic_model(self)

        self.rts_fn = num.rts_fn
        self.h_fn = num.h_fn
        self.h_dot_fn = num.h_dot_fn
        self.h_ddot_fn = num.h_ddot_fn
        self.a_inv_fn = num.a_inv_fn
        self.b_fn = num.b_fn

        return self
