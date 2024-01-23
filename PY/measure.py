import re
import subprocess


def measure():
    suanpan = r"C:\Users\Theodore\Documents\Repo\suanPan\build\Release\suanPan.exe"
    file = r"C:\Users\Theodore\Documents\Repo\nonviscous-implementation\PY\three_dof\three_gauss.sp"

    total = 0

    n = 20

    for i in range(n):
        output = subprocess.check_output([suanpan, '-nc', '-f', file])
        time = re.search(r'Time Wasted: (\d+\.\d+) Seconds', output.decode()).group(1)
        total += float(time)

    print(total / n)


if __name__ == '__main__':
    measure()
