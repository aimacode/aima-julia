include("../aimajulia.jl");

using Base.Test;

using aimajulia;

#The following Logic tests are from the aima-python doctests

A = Expression("A");

B = Expression("B");

C = Expression("C");

D = Expression("D");

E = Expression("E");

F = Expression("F");

G = Expression("G");

H = Expression("H");

x = Expression("x");

y = Expression("y");

z = Expression("z");

P = Expression("P");

Q = Expression("Q");

R = Expression("R");

S = Expression("S");

@test variables(expr("F(x, x) & G(x, y) & H(y, z) & R(A, z, 2)")) == Set(x, y, z);

@test show(expr("P & Q ==> Q")) == "((P & Q) ==> Q)";
