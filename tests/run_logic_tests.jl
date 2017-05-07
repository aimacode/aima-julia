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

@test tt_true("(P ==> Q) <=> (~P | Q)") == true;

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

@test eliminate_implications(expr("A ==> (~B <== C)")) == expr("(~B | ~C) | ~A");

@test eliminate_implications(expr("A ^ B")) == expr("(A & ~B) | (~A & B)");

@test move_not_inwards(expr("~(A | B)")) == expr("~A & ~B");

@test move_not_inwards(expr("~(A & B)")) == expr("~A | ~B");

@test move_not_inwards(expr("~(~(A | ~B) | ~(~C))")) == expr("(A | ~B) & ~C");

@test distribute_and_over_or(expr("(A & B) | C")) == expr("(A | C) & (B | C)");

@test associate("&", (expr("A & B"), expr("B | C"), expr("B & C"))) == expr("&(A, B, (B | C), B, C)");

@test associate("|", (expr("A | (B | (C | (A & B)))"),)) == expr("|(A, B, C, (A & B))");

@test conjuncts(expr("A & B")) == [Expression("A"), Expression("B")];

@test conjuncts(expr("A | B")) == [expr("A | B")];

@test disjuncts(expr("A | B")) == [Expression("A"), Expression("B")];

@test disjuncts(expr("A & B")) == [expr("A & B")];

@test to_conjunctive_normal_form(expr("~(B | C)")) == expr("~B & ~C");

@test repr(to_conjunctive_normal_form(expr("~(B | C)"))) == "(~(B) & ~(C))";

@test to_conjunctive_normal_form(expr("(P & Q) | (~P & ~Q)")) == expr("&((~P | P), (~Q | P), (~P | Q), (~Q | Q))");

@test repr(to_conjunctive_normal_form(expr("(P & Q) | (~P & ~Q)"))) == "((~(P) | P) & (~(Q) | P) & (~(P) | Q) & (~(Q) | Q))";

@test to_conjunctive_normal_form(expr("B <=> (P1 | P2)")) == expr("&((~P1 | B), (~P2 | B), |(P1, P2, ~B))");

@test repr(to_conjunctive_normal_form(expr("B <=> (P1 | P2)"))) == "((~(P1) | B) & (~(P2) | B) & (P1 | P2 | ~(B)))";

@test to_conjunctive_normal_form(expr("a | (b & c) | d")) == expr("|(b, a, d) & |(c, a, d)");

@test repr(to_conjunctive_normal_form(expr("a | (b & c) | d"))) == "((b | a | d) & (c | a | d))";

@test to_conjunctive_normal_form(expr("A & (B | (D & E))")) == expr("&(A, (D | B), (E | B))");

@test repr(to_conjunctive_normal_form(expr("A & (B | (D & E))"))) == "(A & (D | B) & (E | B))";

@test to_conjunctive_normal_form(expr("A | (B | (C | (D & E)))")) == expr("|(D, A, B, C) & |(E, A, B, C)");

@test repr(to_conjunctive_normal_form(expr("A | (B | (C | (D & E)))"))) == "((D | A | B | C) & (E | A | B | C))";

prop_kb = PropositionalKnowledgeBase();

@test count((function(item)
                if (typeof(item) <: Bool)
                    return item;
                else
                    return true;
                end
            end), collect(ask(prop_kb, e) for e in map(expr, ["A", "C", "D", "E", "Q"]))) == 0;

tell(prop_kb, expr("A & E"));

@test ask(prop_kb, expr("A")) == Dict([]);

@test ask(prop_kb, expr("E")) == Dict([]);

tell(prop_kb, expr("E ==> C"));

@test ask(prop_kb, expr("C")) == Dict([]);

retract(prop_kb, expr("E"));

@test ask(prop_kb, expr("E")) == false;

@test ask(prop_kb, expr("C")) == false;

plr_results = pl_resolve(to_conjunctive_normal_form(expr("A | B | C")),
                        to_conjunctive_normal_form(expr("~B | ~C | F")));

@test pretty_set(Set{Expression}(disjuncts(plr_results[1]))) == "Set(aimajulia.Expression[A,C,F,~(C)])";

@test pretty_set(Set{Expression}(disjuncts(plr_results[2]))) == "Set(aimajulia.Expression[A,B,F,~(B)])";

# Use PropositionalKnowledgeBase to represent the Wumpus World (Fig. 7.4)

kb_wumpus = PropositionalKnowledgeBase();
tell(kb_wumpus, expr("~P11"));
tell(kb_wumpus, expr("B11 <=> (P12 | P21)"));
tell(kb_wumpus, expr("B21 <=> (P11 | P22 | P31)"));
tell(kb_wumpus, expr("~B11"));
tell(kb_wumpus, expr("B21"));

# Can't find a pit at location (1, 1).
@test ask(kb_wumpus, expr("~P11")) == Dict([]);

# Can't find a pit at location (1, 2).
@test ask(kb_wumpus, expr("~P12")) == Dict([]);

# Found pit at location (2, 2).
@test ask(kb_wumpus, expr("P22")) == false;

# Found pit at location (3, 1).
@test ask(kb_wumpus, expr("P31")) == false;

# Locations (1, 2) and (2, 1) do not contain pits.
@test ask(kb_wumpus, expr("~P12 & ~P21")) == Dict([]);

# Found a pit in either (3, 1) or (2,2).
@test ask(kb_wumpus, expr("P22 | P31")) == Dict([]);

