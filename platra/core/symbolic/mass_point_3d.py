from sympy import (
    Expr,
    Matrix,
    Rational,
    acos,
    diff,
    hessian,
    lambdify,
    simplify,
    sqrt,
    symbols,
)

from .common import x, y

z, s_dot, m = symbols("z s_dot m")
vx, vy, vz = symbols("vx vy vz")
ax, ay, az = symbols("ax ay az")

phi1 = 0.2 * x**2 + 0.6 * y**2 - 225
phi2 = z + y + 5


def gradient(F: Expr) -> Matrix:
    return Matrix([[diff(F, x), diff(F, y), diff(F, z)]])


def norm(vec) -> Expr:
    return (vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2) ** Rational(1, 2)


def targential(phi1: Expr, phi2: Expr):
    grad1 = gradient(phi1)
    grad2 = gradient(phi2)

    t = grad1.cross(grad2)
    t_norm = norm(t)

    return t, t_norm


def angle(phi1: Expr, phi2: Expr) -> Matrix:
    t, t_norm = targential(phi1, phi2)
    t = t / t_norm

    alpha = acos(t[0] / t_norm)
    beta = acos(t[1] / t_norm)
    gamma = acos(t[2] / t_norm)

    return Matrix([alpha, beta, gamma])


def jacobian(phi1: Expr, phi2: Expr) -> Matrix:
    grad1 = gradient(phi1)
    grad2 = gradient(phi2)

    cross_vec = grad1.cross(grad2)
    cross_norm = norm(cross_vec)

    cross_vec, cross_norm = targential(phi1, phi2)

    norm1 = norm(grad1)
    norm2 = norm(grad2)

    tangent = cross_vec / cross_norm
    normal1 = grad1 / norm1
    normal2 = grad2 / norm2

    return Matrix.vstack(tangent, normal1, normal2)


def curvature(phi1: Expr, phi2: Expr) -> Matrix:
    n1 = gradient(phi1).transpose()
    n2 = gradient(phi2).transpose()

    r_dot = n1.cross(n2)
    r_dot_norm = norm(r_dot)

    if r_dot_norm == 0:
        raise ValueError("Curvature is undefined for zero velocity")

    hess1 = hessian(phi1, (x, y, z))
    hess2 = hessian(phi2, (x, y, z))

    dn1_dt = hess1 * r_dot
    dn2_dt = hess2 * r_dot

    r_ddot = dn1_dt.cross(n2) + n1.cross(dn2_dt)

    numerator_vec = r_dot.cross(r_ddot)
    numerator = numerator_vec.norm()
    denominator = r_dot_norm**3

    k = numerator / denominator

    return simplify(k)


def get_derivative_of_normalized(u: Matrix, u_dot: Matrix) -> Matrix:
    u_inner = u.dot(u)  # |u|^2
    u_dot_inner = u.dot(u_dot)  # u . u'
    u_norm = sqrt(u_inner)  # |u|

    numerator = u_dot * u_inner - u * u_dot_inner
    denominator = u_inner * u_norm  # |u|^3

    return numerator / denominator


def jacobian_derivative(phi1, phi2) -> Matrix:
    grad1 = Matrix([diff(phi1, var) for var in (x, y, z)])
    grad2 = Matrix([diff(phi2, var) for var in (x, y, z)])

    v = grad1.cross(grad2)

    H1 = hessian(phi1, (x, y, z))
    H2 = hessian(phi2, (x, y, z))

    grad1_dot = H1 * v
    grad2_dot = H2 * v

    v_dot = grad1_dot.cross(grad2) + grad1.cross(grad2_dot)

    d_tangent = get_derivative_of_normalized(v, v_dot)
    d_normal1 = get_derivative_of_normalized(grad1, grad1_dot)
    d_normal2 = get_derivative_of_normalized(grad2, grad2_dot)

    return Matrix.hstack(d_tangent, d_normal1, d_normal2)


def omega_star(phi1, phi2):
    xi = curvature(phi1, phi2)
    t, t_norm = targential(phi1, phi2)
    omega_star = xi * s_dot * t / t_norm
    return omega_star


jacobi = jacobian(phi1, phi2)
jacobi_dot = jacobian_derivative(phi1, phi2)
alpha_star = angle(phi1, phi2)
tangent = targential(phi1, phi2)[0]
xi = curvature(phi1, phi2)
omega_star_expr = omega_star(phi1, phi2)


phi1_func = lambdify((x, y, z), phi1, "numpy")
phi2_func = lambdify((x, y, z), phi2, "numpy")

grad_phi1 = gradient(phi1)
grad_phi2 = gradient(phi2)

grad_phi1_func = lambdify((x, y, z), grad_phi1, "numpy")
grad_phi2_func = lambdify((x, y, z), grad_phi2, "numpy")

jacobi_func = lambdify((x, y, z), jacobi, "numpy")
jacobi_dot_func = lambdify((x, y, z), jacobi_dot, "numpy")
alpha_star_func = lambdify((x, y, z), alpha_star, "numpy")
xi_func = lambdify((x, y, z), xi, "numpy")
tangent_func = lambdify((x, y, z), tangent, "numpy")
omega_star_func = lambdify((x, y, z, s_dot), omega_star_expr, "numpy")


class MassPointSymbolic3D:
    @staticmethod
    def phi(index, x, y, z):
        if index == 0:
            return phi1_func(x, y, z)
        elif index == 1:
            return phi2_func(x, y, z)
        else:
            raise ValueError("Неверный индекс функции траектории (должен быть 0 или 1)")

    @staticmethod
    def grad(index, x, y, z):
        if index == 0:
            return grad_phi1_func(x, y, z).reshape(-1)
        elif index == 1:
            return grad_phi2_func(x, y, z).reshape(-1)
        else:
            raise ValueError("Неверный индекс функции траектории (должен быть 0 или 1)")

    @staticmethod
    def jacobi(x, y, z):
        return jacobi_func(x, y, z)

    @staticmethod
    def jacobi_dot(x, y, z):
        return jacobi_dot_func(x, y, z)

    @staticmethod
    def alpha(x, y, z):
        return alpha_star_func(x, y, z).reshape(-1)

    @staticmethod
    def xi(x, y, z):
        return xi_func(x, y, z).reshape(-1)

    @staticmethod
    def tangent(x, y, z):
        return tangent_func(x, y, z).reshape(-1)

    @staticmethod
    def omega_star(x, y, z, s_dot):
        return omega_star_func(x, y, z, s_dot).reshape(-1)
