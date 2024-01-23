# A Strategy for Fast Evaluation of Nonviscously Damped Systems With Arbitrary Kernels

This repository contains the source code and example models of paper [10.1016/j.ymssp.2024.111156](https://doi.org/10.1016/j.ymssp.2024.111156).

To cite or reproduce figures in the paper, you can find the corresponding figure and copy the source code in your work. Please check CI/CD workflow to see how to generate figures used in the paper.

The numerical examples used in the paper are developed in `suanPan`. To perform the numerical analysis, one can download and install [`suanPan`](https://github.com/TLCFEM/suanPan). Then run the model via, for example, the following command in the corresponding
folders under the corresponding folders in `PY`.

```sh
suanpan -f sigmoid.supan
```

The nonviscous damping model is implemented in `suanPan` as:

1. a global damping model with the Newmark method: [NonviscousNewmark](https://tlcfem.github.io/suanPan-manual/latest/Library/Integrator/Newmark/NonviscousNewmark/)
2. a material model that can be used in spring elements: [Nonviscous01](https://tlcfem.github.io/suanPan-manual/latest/Library/Material/Material1D/Viscosity/Nonviscous01/)
3. a modifier that can be applied to arbitrary elements: [ElementalNonviscous](https://tlcfem.github.io/suanPan-manual/latest/Library/Element/Modifier/ElementalNonviscous/)
