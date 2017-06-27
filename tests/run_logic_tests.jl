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

@test repr(to_conjunctive_normal_form(Expression("&",
                                                aimajulia.wumpus_world_inference,
                                                Expression("~", expr("~P12"))))) ==
                                                "((~(P12) | B11) & (~(P21) | B11) & (P12 | P21 | ~(B11)) & ~(B11) & P12)";

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

@test pl_fc_entails(aimajulia.horn_clauses_kb, Expression("Q")) == true;

@test pl_fc_entails(aimajulia.horn_clauses_kb, Expression("SomethingSilly")) == false;

@test inspect_literal(Expression("P")) == (Expression("P"), true);

@test inspect_literal(Expression("~", Expression("P"))) == (Expression("P"), false);

@test unit_clause_assign(expr("A | B | C"), Dict([Pair(Expression("A"), true)])) == (nothing, nothing);

@test unit_clause_assign(expr("B | ~C"), Dict([Pair(Expression("A"), true)])) == (nothing, nothing);

@test unit_clause_assign(expr("B | C"), Dict([Pair(Expression("A"), true)])) == (nothing, nothing);

@test unit_clause_assign(expr("~A | ~B"), Dict([Pair(Expression("A"), true)])) == (Expression("B"), false);

@test unit_clause_assign(expr("B | ~A"), Dict([Pair(Expression("A"), true)])) == (Expression("B"), true);

@test find_unit_clause(map(expr, ["A | B | C", "B | ~C", "~A | ~B"]), Dict([Pair(Expression("A"), true)])) == (Expression("B"), false);

@test find_pure_symbol(map(expr, ["A", "B", "C"]), map(expr, ["A | ~B", "~B | ~C", "C | A"])) == (Expression("A"), true);

@test find_pure_symbol(map(expr, ["A", "B", "C"]), map(expr, ["~A | ~B", "~B | ~C", "C | A"])) == (Expression("B"), false);

@test find_pure_symbol(map(expr, ["A", "B", "C"]), map(expr, ["~A | B", "~B | ~C", "C | A"])) == (nothing, nothing);

@test dpll_satisfiable(expr("A & ~B")) == Dict([Pair(Expression("A"), true),
                                                Pair(Expression("B"), false),]);

@test dpll_satisfiable(expr("P & ~P")) == false;

@test (dpll_satisfiable(expr("A & ~B & C & (A | ~D) & (~E | ~D) & (C | ~D) & (~A | ~F) & (E | ~F) & (~D | ~F) & (B | ~C | D) & (A | ~E | F) & (~A | E | D)"))
        == Dict([Pair(Expression("A"), true),
                Pair(Expression("B"), false),
                Pair(Expression("C"), true),
                Pair(Expression("D"), true),
                Pair(Expression("E"), false),
                Pair(Expression("F"), false),]));

function walksat_test(clauses::Array{Expression, 1}; solutions::Dict=Dict())
    local sln = walksat(clauses);
    if (!(typeof(sln) <: Void)) #found a satisfiable solution
        @test all(collect(pl_true(clause, model=sln) for clause in clauses));
        if (length(solutions) != 0)
            @test all(collect(pl_true(clause, model=solutions) for clause in clauses));
            @test sln == solutions;
        end
    end
    nothing;
end

walksat_test(map(expr, ["A & B", "A & C"]));

walksat_test(map(expr, ["A | B", "P & Q", "P & B"]));

walksat_test(map(expr, ["A & B", "C | D", "~(D | P)"]), solutions=Dict([Pair(Expression("A"), true),
                                                            Pair(Expression("B"), true),
                                                            Pair(Expression("C"), true),
                                                            Pair(Expression("D"), false),
                                                            Pair(Expression("P"), false),]));

@test (typeof(walksat(map(expr, ["A & ~A"]), p=0.5, max_flips=100)) <: Void);

@test (typeof(walksat(map(expr, ["A | B", "~A", "~(B | C)", "C | D", "P | Q"]), p=0.5, max_flips=100)) <: Void);

@test (typeof(walksat(map(expr, ["A | B", "B & C", "C | D", "D & A", "P", "~P"]), p=0.5, max_flips=100)) <: Void);

transition = Dict([Pair("A", Dict([Pair("Left", "A"), Pair("Right", "B")])),
                    Pair("B", Dict([Pair("Left", "A"), Pair("Right", "C")])),
                    Pair("C", Dict([Pair("Left", "B"), Pair("Right", "C")]))]);

@test (typeof(sat_plan("A", transition,"C", 2)) <: Void);

@test sat_plan("A", transition, "B", 3) == ["Right"];

@test sat_plan("C", transition, "A", 3) == ["Left", "Left"];

transition = Dict([Pair((0, 0), Dict([Pair("Right", (0, 1)), Pair("Down", (1, 0))])),
                    Pair((0, 1), Dict([Pair("Left", (1, 0)), Pair("Down", (1, 1))])),
                    Pair((1, 0), Dict([Pair("Right", (1, 0)), Pair("Up", (1, 0)), Pair("Left", (1, 0)), Pair("Down", (1, 0))])),
                    Pair((1, 1), Dict([Pair("Left", (1, 0)), Pair("Up", (0, 1))]))]);

@test sat_plan((0, 0), transition, (1, 1), 4) == ["Right", "Down"];

@test unify(expr("x + y"), expr("y + C"), Dict([])) == Dict([Pair(Expression("x"), Expression("y")),
                                                            Pair(Expression("y"), Expression("C"))]);

@test unify(expr("x"), expr("3"), Dict([])) == Dict([Pair(Expression("x"), Expression("3"))]);

@test unify(expr("x"), expr("x"), Dict([])) == Dict([]);

@test extend(Dict([Pair(Expression("x"), 1)]), Expression("y"), 2) == Dict([Pair(Expression("x"), 1),
                                                                            Pair(Expression("y"), 2)]);

@test repr(substitute(Dict([Pair(Expression("x"), Expression("42")),
                            Pair(Expression("y"), Expression("0"))]),
                    expr("F(x) + y"))) == "(F(42) + 0)";

function fol_bc_ask_query(q::Expression; kb::Union{Void, AbstractKnowledgeBase}=nothing)
    local answers::Tuple;
    if (typeof(kb) <: Void)
        answers = fol_bc_ask(aimajulia.test_fol_kb, q);
    else
        answers = fol_bc_ask(kb, q);
    end
    local test_vars = variables(q);
    return sort(collect(Dict(collect(Pair(k, v) for (k, v) in answer if (k in test_vars))) for answer in answers),
            lt=(function(d1::Dict, d2::Dict)
                    return isless(repr(d1), repr(d2));
                end));
end

@test fol_bc_ask_query(expr("Farmer(x)")) == [Dict([Pair(Expression("x"), Expression("Mac"))])];

@test fol_bc_ask_query(expr("Human(x)")) == [Dict([Pair(Expression("x"), Expression("Mac"))]),
                                            Dict([Pair(Expression("x"), Expression("MrsMac"))])];

@test fol_bc_ask_query(expr("Rabbit(x)")) == [Dict([Pair(Expression("x"), Expression("MrsRabbit"))]),
                                            Dict([Pair(Expression("x"), Expression("Pete"))])];

@test fol_bc_ask_query(expr("Criminal(x)"), kb=aimajulia.crime_kb) == [Dict([Pair(Expression("x"), Expression("West"))])];

@test differentiate(expr("x * x"), expr("x")) == expr("(x * 1) + (x * 1)");

@test differentiate_simplify(expr("x * x"), expr("x")) == expr("2 * x");

