node 1 0 0
node 2 1 0
node 3 2 0
node 4 3 0
node 5 4 0

material Elastic1D 1 2
material Nonviscous01 2 600. 0 1000. 0
material Nonviscous01 3 400. 0 2000. 0

element Spring01 1 1 2 1
element Spring01 2 2 3 1
element Spring01 3 3 4 1
element Spring01 4 4 5 1

mass 5 2 3 1
mass 6 3 3 1
mass 7 4 3 1

element Damper05 8 1 2 2
element Damper05 9 1 3 2
element Damper05 10 3 4 3

fix2 1 1 1 5
fix2 2 2 1 2 3 4 5

initial displacement 1 1 2
initial acceleration -1.333333333333333333333333 1 2
initial acceleration 0.66666666666666666666666666 1 3

hdf5recorder 1 Node U1 2 3 4

step dynamic 1 50
set ini_step_size 5E-2
set fixed_step_size 1

converger AbsIncreDisp 1 1E-14 10 1

analyze

save recorder 1

terminal ren R1-U1.h5 R1-U1-0.05.h5

exit
