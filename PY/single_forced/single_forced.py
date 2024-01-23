import h5py
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from numpy import exp
from scipy.stats import linregress

from PY.Utility import LS, Response

u0 = 0
v0 = 0
a = .4 * np.pi
amp = 100

matplotlib.rcParams.update({'font.size': 6})


def system(m, c, k, nu):
    roots = np.roots([
        m,
        m * nu,
        a ** 2 * m + c * nu + k,
        a ** 2 * m * nu + k * nu,
        a ** 2 * c * nu + a ** 2 * k,
        a ** 2 * k * nu
    ])

    r1, r2, r3, r4, r5 = roots

    coef = np.ones([5, 5], dtype=complex)
    coef[1, 0] = -r2 - r3 - r4 - r5
    coef[2, 0] = r2 * r3 + r2 * r4 + r2 * r5 + r3 * r4 + r3 * r5 + r4 * r5
    coef[3, 0] = -r2 * r3 * r4 - r2 * r3 * r5 - r2 * r4 * r5 - r3 * r4 * r5
    coef[4, 0] = r2 * r3 * r4 * r5

    coef[1, 1] = -r1 - r3 - r4 - r5
    coef[2, 1] = r1 * r3 + r1 * r4 + r1 * r5 + r3 * r4 + r3 * r5 + r4 * r5
    coef[3, 1] = -r1 * r3 * r4 - r1 * r3 * r5 - r1 * r4 * r5 - r3 * r4 * r5
    coef[4, 1] = r1 * r3 * r4 * r5

    coef[1, 2] = -r1 - r2 - r4 - r5
    coef[2, 2] = r1 * r2 + r1 * r4 + r1 * r5 + r2 * r4 + r2 * r5 + r4 * r5
    coef[3, 2] = -r1 * r2 * r4 - r1 * r2 * r5 - r1 * r4 * r5 - r2 * r4 * r5
    coef[4, 2] = r1 * r2 * r4 * r5

    coef[1, 3] = -r1 - r2 - r3 - r5
    coef[2, 3] = r1 * r2 + r1 * r3 + r1 * r5 + r2 * r3 + r2 * r5 + r3 * r5
    coef[3, 3] = -r1 * r2 * r3 - r1 * r2 * r5 - r1 * r3 * r5 - r2 * r3 * r5
    coef[4, 3] = r1 * r2 * r3 * r5

    coef[1, 4] = -r1 - r2 - r3 - r4
    coef[2, 4] = r1 * r2 + r1 * r3 + r1 * r4 + r2 * r3 + r2 * r4 + r3 * r4
    coef[3, 4] = -r1 * r2 * r3 - r1 * r2 * r4 - r1 * r3 * r4 - r2 * r3 * r4
    coef[4, 4] = r1 * r2 * r3 * r4

    w = np.linalg.solve(coef, np.array([
        m * u0,
        m * nu * u0 + m * v0,
        a ** 2 * m * u0 + c * nu * u0 + m * nu * v0,
        a ** 2 * m * nu * u0 + a + a ** 2 * m * v0,
        a ** 2 * m * nu * v0 + a ** 2 * c * nu * u0 + a * nu
    ]))

    def _f(_t):
        return np.dot(w, exp(roots * _t)).real * amp

    return _f


def analytical(para):
    vibrator = system(*para)
    t = 20
    dt = 0.01
    x = np.linspace(0, t, int(t / dt) + 1)
    y = np.zeros(len(x))
    for i in range(len(x)):
        y[i] = vibrator(x[i])

    plt.plot(x, y, 'r', label='analytical', linewidth=.5)

    return vibrator


def numerical(vibrator, pick):
    name = 'R1-U'
    with h5py.File(f'{name}-{str(pick)}.h5', 'r') as f:
        data = f[f'/{name}/{name}2']
        time = data[:, 0]
        displacement = data[:, 1]

    ref = np.zeros(len(time))
    for i in range(len(time)):
        ref[i] = vibrator(time[i])

    error = ref - displacement

    return Response(time, displacement, error)


if __name__ == '__main__':
    fig = plt.figure(figsize=(6, 3.5))
    fig.add_subplot(211)

    results = {}

    sdof = analytical([1, 2, 100, 1])
    results['0.0001'] = numerical(sdof, 0.0001)
    results['0.0002'] = numerical(sdof, 0.0002)
    results['0.0005'] = numerical(sdof, 0.0005)
    results['0.001'] = numerical(sdof, 0.001)
    results['0.002'] = numerical(sdof, 0.002)
    results['0.005'] = numerical(sdof, 0.005)
    results['0.01'] = numerical(sdof, 0.01)

    for key, value in results.items():
        plt.plot(
            value.time, value.displacement, label=f'$\\Delta{{}}t=${float(key):1.0E}', linestyle=next(LS),
            linewidth=.8)

    plt.legend(loc='lower right', ncol=2)
    plt.xlabel('time (s)')
    plt.ylabel('displacement')
    plt.grid(which='both', linestyle='--', linewidth=.2)
    plt.xlim(0, 10)

    fig.add_subplot(2, 1, 2)

    error_x = []
    error_y = []
    for key, value in results.items():
        error_x.append(float(key))
        error_y.append(np.max(np.abs(value.error)))

    if len(error_x) > 1:
        result = linregress(np.log(error_x), np.log(error_y))
        plt.loglog(
            error_x, np.exp(result[1]) * np.power(error_x, result[0]), 'r--',
            label=f'slope {result[0]:.3f} $r^2=${result[2] ** 2:.3f}')
        plt.loglog(error_x, error_y, 'o')
        plt.grid(which='both', linestyle='--', linewidth=.2)
        plt.legend()
        plt.xlabel(r'$\Delta{}t$ (s)')
        plt.ylabel('absolute error $\\epsilon$')

    fig.tight_layout(pad=.1)
    plt.show()
    fig.savefig('../single_forced.pdf')

    fig = plt.figure(figsize=(6, 2))

    for key, value in results.items():
        plt.plot(
            value.time, np.abs(value.error), label=f'$\\Delta{{}}t=${float(key):1.0E}',
            linestyle=next(LS), linewidth=.8)

    plt.yscale('log')
    plt.legend(loc='lower right', ncol=2)
    plt.xlabel('time (s)')
    plt.ylabel('absolute error $\\epsilon$')
    plt.grid(which='both', linestyle='--', linewidth=.2)
    plt.xlim(0, 10)

    fig.tight_layout(pad=.1)
    plt.show()
    fig.savefig('../single_forced_error.pdf')
