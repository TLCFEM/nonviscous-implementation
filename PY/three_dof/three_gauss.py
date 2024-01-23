import h5py
import matplotlib
from matplotlib import pyplot as plt

from PY.Utility import Response
from PY.three_dof.three_dof import add_plot

matplotlib.rcParams.update({'font.size': 6})


def numerical(pick):
    name = 'R1-U1'
    with h5py.File(f'{name}-G{str(pick)}.h5', 'r') as f:
        data1 = f[f'/{name}/{name}2']
        data2 = f[f'/{name}/{name}3']
        data3 = f[f'/{name}/{name}4']
        time = data1[:, 0]
        displacement1 = data1[:, 1]
        displacement2 = data2[:, 1]
        displacement3 = data3[:, 1]

    return (Response(time, displacement1),
            Response(time, displacement2),
            Response(time, displacement3))


def three_dof():
    results = {
        '0.1': numerical(0.1), '0.05': numerical(0.05), '0.02': numerical(0.02), '0.01': numerical(0.01),
        '0.005': numerical(0.005)}

    fig = plt.figure(figsize=(6, 2))

    ax1 = plt.gca()

    add_plot(ax1, results, '0.1', 1.4)
    add_plot(ax1, results, '0.005', 1.4)

    plt.xlabel('time (s)')
    plt.ylabel('displacement')
    plt.grid(which='both', axis='both', linestyle='--', linewidth=.2)
    plt.xlim(0, 30)

    plt.legend(handlelength=4, ncols=2)

    fig.tight_layout(pad=.1)
    plt.show()
    fig.savefig('../three_gauss.pdf')


if __name__ == '__main__':
    three_dof()
