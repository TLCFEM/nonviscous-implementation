################################################################################
#  Copyright (C) 2023 Theodore Chang
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
################################################################################

import re

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

matplotlib.rcParams.update({'font.size': 6})

kernel1 = '''
M = 
+3.1408564644840428e+01+2.4419195061753636e-204j
-1.6706983140997441e+01+1.6944936341891648e+01j
-1.6706983140997441e+01-1.6944936341891648e+01j
+4.3117593746835351e-03+1.0182620511730521e+01j
+4.3117593746835351e-03-1.0182620511730521e+01j
+1.5801217025257626e+00+1.7151347706207849e+00j
+1.5801217025257626e+00-1.7151347706207849e+00j
-2.5283515424182218e-01+3.6580452129545922e-02j
-2.5283515424182218e-01-3.6580452129545922e-02j
+9.6864499480377557e-03+4.3130572825140741e-03j
+9.6864499480377557e-03-4.3130572825140741e-03j
-7.0188900179621449e-05+5.9181046455302063e-05j
-7.0188900179621449e-05-5.9181046455302063e-05j
S = 
+4.5016919366209596e+00-0.0000000000000000e+00j
+4.4950837272609609e+00+1.0395441820843294e+00j
+4.4950837272609609e+00-1.0395441820843294e+00j
+4.4749101704334056e+00-2.0949342266104529e+00j
+4.4749101704334056e+00+2.0949342266104529e+00j
+4.4400143885590264e+00+3.1851640805978376e+00j
+4.4400143885590264e+00-3.1851640805978376e+00j
+4.3879941466299011e+00-4.3376514748574726e+00j
+4.3879941466299011e+00+4.3376514748574726e+00j
+4.3138681929799541e+00-5.6015344176775130e+00j
+4.3138681929799541e+00+5.6015344176775130e+00j
+4.2046027378649198e+00+7.1006004448588689e+00j
+4.2046027378649198e+00-7.1006004448588689e+00j
'''
kernel2 = '''
M = 
+2.3906159000727541e+01+1.3867518867778434e-312j
-1.2762256426897499e+01+1.2860476490373076e+01j
-1.2762256426897499e+01-1.2860476490373076e+01j
+5.3218297629940091e-02+7.7608889955930351e+00j
+5.3218297629940091e-02-7.7608889955930351e+00j
+1.1940946465978099e+00+1.3211005737652248e+00j
+1.1940946465978099e+00-1.3211005737652248e+00j
-1.9326268560869156e-01+3.0383779818506866e-02j
-1.9326268560869156e-01-3.0383779818506866e-02j
+7.4948577453368154e-03+3.2098505412941876e-03j
+7.4948577453368154e-03-3.2098505412941876e-03j
-5.4937628189936578e-05+4.4960570107889509e-05j
-5.4937628189936578e-05-4.4960570107889509e-05j
S = 
+1.0087637748351039e+01-0.0000000000000000e+00j
+1.0072828830375100e+01+2.3227012589928293e+00j
+1.0072828830375100e+01-2.3227012589928293e+00j
+1.0027597533472361e+01-4.6807376205102571e+00j
+1.0027597533472361e+01+4.6807376205102571e+00j
+9.9492752852218480e+00+7.1164847349390277e+00j
+9.9492752852218480e+00-7.1164847349390277e+00j
+9.8323245672882003e+00-9.6911681681668096e+00j
+9.8323245672882003e+00+9.6911681681668096e+00j
+9.6653057328147884e+00-1.2514633249962301e+01j
+9.6653057328147884e+00+1.2514633249962301e+01j
+9.4185072574984083e+00+1.5863782488080192e+01j
+9.4185072574984083e+00-1.5863782488080192e+01j
'''


# change this kernel before plotting
def kernel1_ana(x):
    return 1.2 * np.sqrt(1 / np.pi) * np.exp(-x ** 2)


def kernel2_ana(x):
    return .4 * np.sqrt(5 / np.pi) * np.exp(-5 * x ** 2)


def split(r: str):
    split_r = r.strip().split('\n')
    regex = re.compile(r'([+\-]\d+\.\d+e[+\-]\d+){2}j')
    items = [i for i in split_r if regex.match(i)]
    if len(items) % 2 != 0:
        print('something wrong with the output')
        return None

    m_complex = [complex(i) for i in items[:len(items) // 2]]
    s_complex = [complex(i) for i in items[len(items) // 2:]]
    return np.array(m_complex), np.array(s_complex)


def print_table(m, s):
    print(r'\begin{tabular}{r|r|r|r}')
    print(r'\toprule')
    print(r'$\Re(m)$ & $\Im(m)$ & $\Re(s)$ & $\Im(s)$ \\')
    print(r'\midrule')

    for i, j in zip(m, s):
        print(f'\\num{{{i.real:.15e}}}&\\num{{{i.imag:.15e}}}&', end='')
        print(f'\\num{{{j.real:.15e}}}&\\num{{{j.imag:.15e}}}\\\\')

    print(r'\bottomrule')
    print(r'\end{tabular}')

    print('================================================================================================')

    for i, j in zip(m, s):
        print(f'{i.real: .15e} {i.imag: .15e} {j.real: .15e} {j.imag: .15e} \\')

    print('================================================================================================')


def plotter(output: str, k):
    if (result := split(output)) is None:
        return

    print_table(*result)

    x = np.linspace(0, 5, 501)
    yy = k(x)
    y = np.zeros(len(x), dtype=complex)
    for ml, sl in zip(*result):
        y += ml * np.exp(-sl * x)

    return x, y, yy


if __name__ == '__main__':
    fig, axs = plt.subplots(2, 1, figsize=(6, 4))
    x, y, ref = plotter(kernel1, kernel1_ana)

    axs[0].plot(x, ref, 'b-', label='kernel $g_1$', linewidth=2)
    axs[0].plot(x, y.real, 'r', linestyle='dashdot', label='approximation', linewidth=3)
    axs[0].legend(handlelength=6)

    ax2 = axs[0].twinx()
    ax2.plot(x, np.abs(ref - y), 'g--', label='absolute error', linewidth=1)
    ax2.set_yscale('log')
    ax2.legend(loc='center right', handlelength=6)

    x, y, ref = plotter(kernel2, kernel2_ana)

    axs[1].plot(x, ref, 'b-', label='kernel $g_2$', linewidth=2)
    axs[1].plot(x, y.real, 'r', linestyle='dashdot', label='approximation', linewidth=3)
    axs[1].legend(handlelength=6)

    ax4 = axs[1].twinx()
    ax4.plot(x, np.abs(ref - y), 'g--', label='absolute error', linewidth=1)
    ax4.set_yscale('log')
    ax4.legend(loc='center right', handlelength=6)

    ax4.set_ylim(1e-15, 1e-12)
    ax2.sharey(ax4)
    ax4.set_ylabel('absolute error')
    ax2.set_ylabel('absolute error')

    axs[0].set_ylabel('kernel $g_1(t)$')
    axs[1].set_ylabel('kernel $g_2(t)$')
    axs[1].set_xlabel('time $t$ (s)')

    plt.setp(axs, xlim=(0, 3))
    plt.tight_layout(pad=.05)
    plt.show()
    fig.savefig('../kernel.pdf')
