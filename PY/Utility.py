from itertools import cycle


def get_line_style():
    ls_tuple = [
        ('solid', (0, ())),
        ('loosely dotted', (0, (1, 4))),
        ('dotted', (0, (1, 2))),
        ('densely dotted', (0, (1, 1))),

        ('loosely dashed', (0, (5, 4))),
        ('dashed', (0, (5, 2))),
        ('densely dashed', (0, (5, 1))),

        ('loosely dashdotted', (0, (3, 4, 1, 4))),
        ('dashdotted', (0, (3, 2, 1, 2))),
        ('densely dashdotted', (0, (3, 1, 1, 1))),

        ('loosely dashdotdotted', (0, (3, 4, 1, 4, 1, 4))),
        ('dashdotdotted', (0, (3, 2, 1, 2, 1, 2))),
        ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))
    ]

    for v in cycle(ls_tuple):
        yield v[1]


LS = get_line_style()
