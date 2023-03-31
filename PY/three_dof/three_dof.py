import h5py
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress

from PY.Utility import LS, Response

matplotlib.rcParams.update({'font.size': 6})


def modal_expansion():
    m = 3
    k = 2
    c1 = .6
    c2 = .2
    mu1 = 1
    mu2 = 5

    A = np.zeros([9, 9])
    B = np.zeros([9, 9])

    A[0, 0] = -2 * k
    A[1, 1] = -2 * k
    A[2, 2] = -2 * k
    A[0, 1] = k
    A[1, 0] = k
    A[1, 2] = k
    A[2, 1] = k

    A[3, 3] = m
    A[4, 4] = m
    A[5, 5] = m

    A[6, 6] = -c1 / mu1
    A[7, 7] = -c1 / mu1
    A[8, 8] = -2 * c2 / mu2

    B[0, 0] = c1
    B[1, 1] = c1 + c2
    B[2, 2] = c2
    B[1, 2] = -c2
    B[2, 1] = -c2

    B[3, 0] = m
    B[4, 1] = m
    B[5, 2] = m

    B[0, 3] = m
    B[1, 4] = m
    B[2, 5] = m

    B[6, 6] = c1 / mu1 / mu1
    B[7, 7] = c1 / mu1 / mu1
    B[8, 8] = 2 * c2 / mu2 / mu2

    B[0, 7] = -c1 / mu1
    B[1, 6] = -c1 / mu1

    B[7, 0] = -c1 / mu1
    B[6, 1] = -c1 / mu1

    B[1, 8] = c2 * np.sqrt(2) / mu2
    B[2, 8] = -c2 * np.sqrt(2) / mu2

    B[8, 1] = c2 * np.sqrt(2) / mu2
    B[8, 2] = -c2 * np.sqrt(2) / mu2

    eig_val, eig_vec = np.linalg.eig(np.linalg.solve(B, A))

    ic = np.zeros(9)

    ic[0] = 1

    c = np.linalg.solve(eig_vec, ic)

    def _f(_t):
        val = np.matmul(eig_vec, np.multiply(c, np.exp(eig_val * _t)))
        return val[0].real, val[1].real, val[2].real

    return _f


def analytical(system):
    t = 50
    dt = 0.01
    x = np.linspace(0, t, int(t / dt) + 1)
    y1 = np.zeros(len(x))
    y2 = np.zeros(len(x))
    y3 = np.zeros(len(x))
    for i in range(len(x)):
        y1[i], y2[i], y3[i] = system(x[i])

    return x, y1, y2, y3


def numerical(vibrator, pick):
    name = 'R1-U1'
    with h5py.File(f'{name}-{str(pick)}.h5', 'r') as f:
        data1 = f[f'/{name}/{name}2']
        data2 = f[f'/{name}/{name}3']
        data3 = f[f'/{name}/{name}4']
        time = data1[:, 0]
        displacement1 = data1[:, 1]
        displacement2 = data2[:, 1]
        displacement3 = data3[:, 1]

    ref1 = np.zeros(len(time))
    ref2 = np.zeros(len(time))
    ref3 = np.zeros(len(time))
    for i in range(len(time)):
        ref1[i], ref2[i], ref3[i] = vibrator(time[i])

    return (Response(time, displacement1, ref1 - displacement1),
            Response(time, displacement2, ref2 - displacement2),
            Response(time, displacement3, ref3 - displacement3))


def add_plot(ax, results, pick):
    response = results[pick]
    ax.plot(
        response[0].time, response[0].displacement, label=f'$x_1$ ($\\Delta{{}}t={pick}$)', linewidth=2,
        linestyle=next(LS))
    ax.plot(
        response[1].time, response[1].displacement, label=f'$x_2$ ($\\Delta{{}}t={pick}$)', linewidth=2,
        linestyle=next(LS))
    ax.plot(
        response[2].time, response[2].displacement, label=f'$x_3$ ($\\Delta{{}}t={pick}$)', linewidth=2,
        linestyle=next(LS))


def three_dof():
    system = modal_expansion()

    results = {}

    results['0.1'] = numerical(system, 0.1)
    results['0.05'] = numerical(system, 0.05)
    results['0.02'] = numerical(system, 0.02)
    results['0.01'] = numerical(system, 0.01)
    results['0.005'] = numerical(system, 0.005)
    results['0.002'] = numerical(system, 0.002)
    results['0.001'] = numerical(system, 0.001)

    fig = plt.figure(figsize=(6, 3.5))

    ax1 = fig.add_subplot(2, 1, 1)
    x, y1, y2, y3 = analytical(system)

    ax1.plot(x, y1, label='$x_1$ analytical', linewidth=1)
    ax1.plot(x, y2, label='$x_2$ analytical', linewidth=1)
    ax1.plot(x, y3, label='$x_3$ analytical', linewidth=1)
    next(LS)

    add_plot(ax1, results, '0.1')
    add_plot(ax1, results, '0.05')

    ax1.legend(ncol=3, loc='upper right')

    plt.xlabel('time (s)')
    plt.ylabel('displacement')
    plt.grid(which='both', axis='both', linestyle='--', linewidth=.2)
    plt.xlim(0, 50)

    fig.add_subplot(2, 1, 2)

    error_x = []
    error_y1 = []
    error_y2 = []
    error_y3 = []
    for key, value in results.items():
        error_x.append(float(key))
        error_y1.append(np.mean(np.abs(value[0].error)))
        error_y2.append(np.mean(np.abs(value[1].error)))
        error_y3.append(np.mean(np.abs(value[2].error)))

    for data, key in zip([error_y1, error_y2, error_y3], ['$x_1$', '$x_2$', '$x_3$']):
        result = linregress(np.log(error_x), np.log(data))
        plt.loglog(error_x, np.exp(result[1]) * np.power(error_x, result[0]),
                   label=f'{key} slope {result[0]:.3f} $r^2=${result[2] ** 2:.3f}')
        plt.loglog(error_x, data, 'o')

    plt.grid(which='both', linestyle='--', linewidth=.2)
    plt.legend()
    plt.xlabel(r'$\Delta{}t$ (s)')
    plt.ylabel('absolute error $\\epsilon$')

    fig.tight_layout(pad=.1)
    plt.show()
    fig.savefig('../three_dof.pdf')


if __name__ == '__main__':
    three_dof()
