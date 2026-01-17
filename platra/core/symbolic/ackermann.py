import logging
from typing import Callable

from sympy import (
    Matrix,
    cos,
    diff,
    lambdify,
    pi,
    sin,
    sqrt,
    symbols,
)

from ..robot.ackermann import AckermannState, AckermannStateExt
from ..robot.configs import AckermannConfigForStaticFeedback
from .common import nu, x, y

logger = logging.getLogger(__name__)

xsi = Matrix([x, y, nu])
R = Matrix([[cos(nu), sin(nu), 0], [-sin(nu), cos(nu), 0], [0, 0, 1]])

alpha_3s, beta_3s, Ls, Lf, e = symbols("alpha_3s beta_3s L_s L_f e")
eta, zeta = symbols("eta, zeta")
v1, v2 = symbols("v_1 v_2")
chi, w = symbols("chi omega")

ls = sqrt(Lf**2 + Ls**2)

Sigma = Matrix([0, ls * sin(beta_3s), -sin(alpha_3s + beta_3s)])
RTS = R.T * Sigma


class LambdifiedAckermannForStaticFeedback:
    def __init__(self, conf: AckermannConfigForStaticFeedback) -> None:
        logger.debug(f"Initializing {self.__repr__()}")

        h = Matrix(
            [
                x
                + ls * sin(alpha_3s + pi / 2 + nu)
                - e * cos(-pi / 2 - alpha_3s - beta_3s - nu),
                y
                - ls * cos(alpha_3s + pi / 2 + nu)
                + e * sin(-pi / 2 - alpha_3s - beta_3s - nu),
            ]
        )
        dh_dx = h.diff(x)
        dh_dy = h.diff(y)
        dh_dnu = h.diff(nu)
        dh_dxsi = Matrix([[dh_dx, dh_dy, dh_dnu]])
        dh_dbeta = h.diff(beta_3s)

        K = Matrix([[dh_dxsi * RTS, dh_dbeta]])
        K_inv = K.inv()
        Kez = K * Matrix([eta, zeta])
        g = (
            Matrix([[Kez.diff(x), Kez.diff(y), Kez.diff(nu)]]) * RTS * eta
            + Kez.diff(beta_3s) * zeta
        )

        # Вывод результатов
        # print("=== Для робота типа (1,1) с динамической позиционной моделью ===")
        # print("Линеаризующие выходы h:")
        # print(latex(h))
        #
        # print("Матрица K")
        # print(latex(K))
        #
        # print("Вектор g")
        # print(latex(g))

        params = {Ls: conf.Ls, Lf: conf.Lf, e: conf.e, alpha_3s: conf.alpha3s}

        self._rts_subed = RTS.subs(params)
        self._h_subed = h.subs(params)
        self._k_inv_subed = K_inv.subs(params)
        self._g_subed = g.subs(params)

        self.rts_lambdified = lambdify([x, y, nu, beta_3s], self._rts_subed, "numpy")
        self.h_lambdified = lambdify([x, y, nu, beta_3s], self._h_subed, "numpy")
        self.k_inv_lambdified = lambdify(
            [x, y, nu, beta_3s], self._k_inv_subed, "numpy"
        )
        self.g_lambdified = lambdify(
            [x, y, nu, beta_3s, eta, zeta], self._g_subed, "numpy"
        )

        logger.debug(f"Initialized {self.__repr__()}")

    def rts_fn(self, state: AckermannState) -> Callable:
        return self.rts_lambdified(
            state.xsi[0], state.xsi[1], state.xsi[2], state.beta_s
        ).reshape(-1)

    def h_fn(self, state: AckermannState) -> Callable:
        return self.h_lambdified(
            state.xsi[0], state.xsi[1], state.xsi[2], state.beta_s
        ).reshape(-1)

    def k_inv_fn(self, state: AckermannState) -> Callable:
        return self.k_inv_lambdified(
            state.xsi[0], state.xsi[1], state.xsi[2], state.beta_s
        )

    def g_fn(self, state: AckermannState) -> Callable:
        return self.g_lambdified(
            state.xsi[0],
            state.xsi[1],
            state.xsi[2],
            state.beta_s,
            state.eta,
            state.zeta,
        ).reshape(-1)


class LambdifiedAckermannForDynamicFeedback:
    def __init__(self, conf: AckermannConfigForStaticFeedback) -> None:
        U1 = symbols("U_1")

        dxsi_dt = RTS * eta
        dx_dt = dxsi_dt[0]
        dy_dt = dxsi_dt[1]
        dnu_dt = dxsi_dt[2]
        dbeta_s_dt = zeta
        deta_dt = v1
        dzeta_dt = v2
        dv1_dt = U1

        h = Matrix([x + e * sin(nu), y - e * cos(nu)])

        h1_dot = diff(h[0], x) * dx_dt + diff(h[0], y) * dy_dt + diff(h[0], nu) * dnu_dt
        h2_dot = diff(h[1], x) * dx_dt + diff(h[1], y) * dy_dt + diff(h[1], nu) * dnu_dt
        h_dot = Matrix([h1_dot, h2_dot])

        h1_ddot = (
            diff(h1_dot, x) * dx_dt
            + diff(h1_dot, y) * dy_dt
            + diff(h1_dot, nu) * dnu_dt
            + diff(h1_dot, eta) * deta_dt
            + diff(h1_dot, zeta) * dzeta_dt
            + diff(h1_dot, beta_3s) * dbeta_s_dt
        )
        h2_ddot = (
            diff(h2_dot, x) * dx_dt
            + diff(h2_dot, y) * dy_dt
            + diff(h2_dot, nu) * dnu_dt
            + diff(h2_dot, eta) * deta_dt
            + diff(h2_dot, zeta) * dzeta_dt
            + diff(h2_dot, beta_3s) * dbeta_s_dt
        )
        h_ddot = Matrix([h1_ddot, h2_ddot])

        h1_dddot = (
            diff(h1_ddot, x) * dx_dt
            + diff(h1_ddot, y) * dy_dt
            + diff(h1_ddot, nu) * dnu_dt
            + diff(h1_ddot, eta) * deta_dt
            + diff(h1_ddot, zeta) * dzeta_dt
            + diff(h1_ddot, beta_3s) * dbeta_s_dt
            + diff(h1_ddot, v1) * dv1_dt
        )
        h2_dddot = (
            diff(h2_ddot, x) * dx_dt
            + diff(h2_ddot, y) * dy_dt
            + diff(h2_ddot, nu) * dnu_dt
            + diff(h2_ddot, eta) * deta_dt
            + diff(h2_ddot, zeta) * dzeta_dt
            + diff(h2_ddot, beta_3s) * dbeta_s_dt
            + diff(h2_ddot, v1) * dv1_dt
        )
        h_dddot = Matrix([h1_dddot, h2_dddot])

        U_vec = Matrix([U1, v2])

        A = Matrix(
            [
                [diff(h_dddot[0], U1), diff(h_dddot[0], v2)],
                [diff(h_dddot[1], U1), diff(h_dddot[1], v2)],
            ]
        )
        b = h_dddot - A * U_vec

        # Вывод результатов
        # print("\nh")
        # pprint(h)
        # print("h_dot")
        # pprint(h_dot.subs(ls, symbols("l_s")))
        # print("h_ddot")
        # pprint(h_ddot.subs(ls, symbols("l_s")))
        # print("h_dddot")
        # pprint(h_dddot.subs(ls, symbols("l_s")))
        #
        # print("\nA:")
        # pprint(A)
        #
        # print("\nb:")
        # pprint(b)

        params = {Ls: conf.Ls, Lf: conf.Lf, e: conf.e, alpha_3s: conf.alpha3s}

        self._rts_subed = RTS.subs(params)
        self._h_subed = h.subs(params)
        self._h_dot_subed = h_dot.subs(params)
        self._h_ddot_subed = h_ddot.subs(params)
        self._a_inv_subed = A.inv().subs(params)
        self._b_subed = b.subs(params)

        self.rts_lambdified = lambdify([x, y, nu, beta_3s], self._rts_subed, "numpy")
        self.h_lambdified = lambdify([x, y, nu, beta_3s], self._h_subed, "numpy")
        self.h_dot_lambdified = lambdify([nu, beta_3s, eta], self._h_dot_subed, "numpy")
        self.h_ddot_lambdified = lambdify(
            [nu, beta_3s, eta, zeta, v1], self._h_ddot_subed, "numpy"
        )
        self.a_inv_lambdified = lambdify([nu, beta_3s, eta], self._a_inv_subed, "numpy")
        self.b_lambdified = lambdify(
            [nu, beta_3s, eta, zeta, v1], self._b_subed, "numpy"
        )

    def rts_fn(self, s: AckermannState) -> Callable:
        return self.rts_lambdified(s.x, s.y, s.nu, s.beta_s).reshape(-1)

    def h_fn(self, s: AckermannState) -> Callable:
        return self.h_lambdified(s.x, s.y, s.nu, s.beta_s).reshape(-1)

    def h_dot_fn(self, s: AckermannState) -> Callable:
        return self.h_dot_lambdified(s.nu, s.beta_s, s.eta).reshape(-1)

    def h_ddot_fn(self, s: AckermannStateExt) -> Callable:
        return self.h_ddot_lambdified(s.nu, s.beta_s, s.eta, s.zeta, s.v1).reshape(-1)

    def a_inv_fn(self, s: AckermannState) -> Callable:
        return self.a_inv_lambdified(s.nu, s.beta_s, s.eta)

    def b_fn(self, s: AckermannStateExt) -> Callable:
        return self.b_lambdified(s.nu, s.beta_s, s.eta, s.zeta, s.v1).reshape(-1)
