include("../aimajulia.jl");

using Base.Test;

using aimajulia;

#The following Logic tests are from the aima-python doctests

x = Expression("x");

y = Expression("y");

z = Expression("z");

@test variables(expr("F(x, x) & G(x, y) & H(y, z) & R(A, z, z)")) == Set(x, y, z);

@test variables(expr("F(x, A, y)")) == Set(x, y);

@test variables(expr("F(G(x), z)")) == Set(x, z);

@test show(expr("P & Q ==> Q")) == "((P & Q) ==> Q)";

@test show(expr("P ==> Q(1)")) == "(P ==> Q(1))";

@test show(expr("P & Q | ~R(x, F(x))")) == "((P & Q) | ~(R(x, F(x))))";

@test show(expr("P & Q ==> R & S")) == "(((P & Q) ==> R) & S)";

