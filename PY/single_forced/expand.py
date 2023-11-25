from sympy import Symbol, latex


def run():
    m = Symbol('m')
    s = Symbol('s')
    U = Symbol('U')
    u0 = Symbol('u0')
    v0 = Symbol('v0')
    mu = Symbol('mu')
    a = Symbol('a')
    c = Symbol('c')
    k = Symbol('k')

    eq = m * (s ** 2 * U - s * u0 - v0)
    eq += c * mu * (s * U - u0) / (s + mu)
    eq += k * U
    eq *= (s ** 2 + a ** 2) * (s + mu)
    eq -= a * (s + mu)

    eq = eq.expand().simplify()

    print(latex(eq))


def term():
    r2 = Symbol('r2')
    r3 = Symbol('r3')
    r4 = Symbol('r4')
    r5 = Symbol('r5')
    s = Symbol('s')

    eq = (s - r2) * (s - r3) * (s - r4) * (s - r5)
    eq = eq.expand().simplify()
    print(latex(eq))


if __name__ == '__main__':
    run()
    term()
