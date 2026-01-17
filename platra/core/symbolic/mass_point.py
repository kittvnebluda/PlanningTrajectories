from sympy import (
    Expr,
    Rational,
    atan2,
    cos,
    diff,
    lambdify,
    simplify,
    sin,
    sqrt,
    symbols,
)

from .common import x, y

r, theta = symbols("r theta", real=True)

CURVES_LAB3 = [
    -1.3 * sin(3 * x + 0.9) + y + 2,
    (x - 10) ** 2 + (y - 3) ** 2 - 36,
    -2.5 * cos(1.2 * x - 0.3) + y - 7,
]

EUCLIDEAN_SPIRAL = [-r + theta]


def xi(F: Expr) -> Expr:
    Fx = diff(F, x)
    Fy = diff(F, y)

    Fxx = diff(Fx, x)
    Fyy = diff(Fy, y)
    Fxy = diff(Fx, y)

    numerator = Fxx * Fy**2 - 2 * Fxy * Fx * Fy + Fyy * Fx**2
    denominator = (Fx**2 + Fy**2) ** Rational(3, 2)

    return simplify(numerator / denominator)


def angle(F: Expr) -> Expr:
    return atan2(F.diff(x), -F.diff(y))


class PolarImplicitCurve:
    def __init__(self, F_polar: Expr):
        self.F_polar = simplify(F_polar)

        self.F_cart = simplify(
            self.F_polar.subs({r: sqrt(x**2 + y**2), theta: atan2(y, x)})
        )

        self._xi = xi(self.F_cart)
        self._alpha = angle(self.F_cart)
        self._xi_dot = self._xi.diff(x) + self._xi.diff(y)

        self.phi_func = lambdify((x, y), self.F_cart, "numpy")
        self.xi_func = lambdify((x, y), self._xi, "numpy")
        self.alpha_func = lambdify((x, y), self._alpha, "numpy")
        self.xi_dot_funcs = lambdify((x, y), self._xi_dot, "numpy")

    def phi(self, x, y):
        return self.phi_func(x, y)

    def xi(self, x, y):
        return self.xi_func(x, y)

    def alpha(self, x, y):
        return self.alpha_func(x, y)

    def xi_dot(self, x, y):
        return self.xi_dot_funcs(x, y)


class MassPointSymbolic:
    def __init__(self, curves) -> None:
        alphas = [angle(f) for f in curves]
        xis = [xi(f) for f in curves]
        xis_dot = [xi.diff(x) + xi.diff(y) for xi in xis]

        self.phi_funcs = [lambdify((x, y), f, "numpy") for f in curves]
        self.alpha_funcs = [lambdify((x, y), f, "numpy") for f in alphas]
        self.xi_funcs = [lambdify((x, y), f, "numpy") for f in xis]
        self.xi_dot_funcs = [lambdify((x, y), f, "numpy") for f in xis_dot]

    def phi(self, index, x, y):
        return self.phi_funcs[index](x, y)

    def xi(self, index, x, y):
        return self.xi_funcs[index](x, y)

    def alpha(self, index, x, y):
        return self.alpha_funcs[index](x, y)

    def xi_dot(self, index, x, y):
        return self.xi_dot_funcs[index](x, y)
