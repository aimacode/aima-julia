include("../aimajulia.jl");

using Base.Test;

using aimajulia;

#The following Logic tests are from the aima-python doctests

x = Expression("x");

y = Expression("y");

z = Expression("z");

@test variables(expr("F(x, x) & G(x, y) & H(y, z) & R(A, z, z)")) == Set([x, y, z]);

@test variables(expr("F(x, A, y)")) == Set([x, y]);

@test variables(expr("F(G(x), z)")) == Set([x, z]);

@test show(expr("P & Q ==> Q")) == "((P & Q) ==> Q)";

@test show(expr("P ==> Q(1)")) == "(P ==> Q(1))";

@test show(expr("P & Q | ~R(x, F(x))")) == "((P & Q) | ~(R(x, F(x))))";

@test show(expr("P & Q ==> R & S")) == "(((P & Q) ==> R) & S)";

@test tt_entails(expr("P & Q"), expr("Q")) == true;

@test tt_entails(expr("P | Q"), expr("Q")) == false;

@test tt_entails(expr("A & (B | C) & E & F & ~(P | Q)"), expr("A & E & F & ~P & ~Q")) == true;

@test proposition_symbols(expr("x & y & z | A")) == [Expression("A")];

@test proposition_symbols(expr("(x & B(z)) ==> Farmer(y) | A")) == [Expression("A"), expr("Farmer(y)"), expr("B(z)")];

@test typeof(pl_true(Expression("P"))) <: Void;

@test typeof(pl_true(expr("P | P"))) <: Void;

@test pl_true(expr("P | Q"), model=Dict([Pair(Expression("P"), true)])) == true;

@test pl_true(expr("(A | B) & (C | D)"),
                model=Dict([Pair(Expression("A"), false),
                            Pair(Expression("B"), true),
                            Pair(Expression("C"), true)])) == true;

@test pl_true(expr("(A & B) & (C | D)"),
                model=Dict([Pair(Expression("A"), false),
                            Pair(Expression("B"), true),
                            Pair(Expression("C"), true)])) == false;

@test pl_true(expr("(A & B) | (A & C)"),
                model=Dict([Pair(Expression("A"), false),
                            Pair(Expression("B"), true),
                            Pair(Expression("C"), true)])) == false;

@test typeof(pl_true(expr("(A | B) & (C | D)"),
                model=Dict([Pair(Expression("A"), true),
                            Pair(Expression("D"), false)]))) <: Void;

@test pl_true(Expression("P"), model=Dict([Pair(Expression("P"), false)])) == false;

@test typeof(pl_true(expr("P | ~P"))) <: Void;

