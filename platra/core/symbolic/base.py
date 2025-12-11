from sympy import Matrix, cos, sin, symbols

x, y, nu = symbols("x y nu")

xsi = Matrix([x, y, nu])
R = Matrix([[cos(nu), sin(nu), 0], [-sin(nu), cos(nu), 0], [0, 0, 1]])
