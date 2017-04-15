include("../aimajulia.jl");

using Base.Test;

using aimajulia;

#The following CSP tests are from the aima-python doctest

@test (solution(depth_first_graph_search(aimajulia.australia_csp))...) == (("NSW","B"),("Q","G"),("NT","B"),("T","B"),("V","G"),("SA","R"),("WA","G"));

d = CSPDict(ConstantFunctionDict(42));

@test d["life"] == 42;

@test (!(typeof(backtracking_search(aimajulia.australia_csp)) <: Void) == true);

@test (!(typeof(backtracking_search(aimajulia.australia_csp,
                                    select_unassigned_variable=minimum_remaining_values)) <: Void) == true);

@test (!(typeof(backtracking_search(aimajulia.australia_csp,
                                    order_domain_values=least_constraining_values)) <: Void) == true);

@test (!(typeof(backtracking_search(aimajulia.australia_csp,
                                    select_unassigned_variable=minimum_remaining_values,
                                    order_domain_values=least_constraining_values)) <: Void) == true);

@test (!(typeof(backtracking_search(aimajulia.australia_csp, inference=forward_checking)) <: Void) == true);

@test (!(typeof(backtracking_search(aimajulia.australia_csp,
                                    inference=maintain_arc_consistency)) <: Void) == true);

@test (!(typeof(backtracking_search(aimajulia.australia_csp,
                                    select_unassigned_variable=minimum_remaining_values,
                                    order_domain_values=least_constraining_values,
                                    inference=maintain_arc_consistency)) <: Void) == true);

topological_sorted_nodes, parent_dict = topological_sort(aimajulia.australia_csp, "NT");

@test topological_sorted_nodes == Any["NT","SA","Q","NSW","V","WA"];

@test haskey(parent_dict, "NT") == false;

@test parent_dict == Dict{Any,Any}(Pair("NSW","Q"), Pair("Q","SA"), Pair("V","NSW"), Pair("SA","NT"), Pair("WA","SA"));

@test length(backtracking_search(NQueensCSP(8))) == 8;

e = SudokuCSP(aimajulia.easy_sudoku_grid);

@test display(e, infer_assignment(e)) == ". . 3 | . 2 . | 6 . .\n9 . . | 3 . 5 | . . 1\n. . 1 | 8 . 6 | 4 . .\n------+-------+------\n. . 8 | 1 . 2 | 9 . .\n7 . . | . . . | . . 8\n. . 6 | 7 . 8 | 2 . .\n------+-------+------\n. . 2 | 6 . 9 | 5 . .\n8 . . | 2 . 3 | . . 9\n. . 5 | . 1 . | 3 . .";

AC3(e);

@test display(e, infer_assignment(e)) == "4 8 3 | 9 2 1 | 6 5 7\n9 6 7 | 3 4 5 | 8 2 1\n2 5 1 | 8 7 6 | 4 9 3\n------+-------+------\n5 4 8 | 1 3 2 | 9 7 6\n7 2 9 | 5 6 4 | 1 3 8\n1 3 6 | 7 9 8 | 2 4 5\n------+-------+------\n3 7 2 | 6 8 9 | 5 1 4\n8 1 4 | 2 5 3 | 7 6 9\n6 9 5 | 4 1 7 | 3 8 2";

@test !(typeof(backtracking_search(SudokuCSP(aimajulia.harder_sudoku_grid), select_unassigned_variable=minimum_remaining_values, inference=forward_checking)) <: Void)

@test solve_zebra(ZebraCSP(), backtracking_search) == (5,
                                                        1,
                                                        75472,
                                                        Dict{Any,Any}(Pair{Any,Any}("Tea",2),
                                                                    Pair{Any,Any}("Red",3),
                                                                    Pair{Any,Any}("Kools",1),
                                                                    Pair{Any,Any}("Green",5),
                                                                    Pair{Any,Any}("Horse",2),
                                                                    Pair{Any,Any}("Zebra",5),
                                                                    Pair{Any,Any}("OJ",4),
                                                                    Pair{Any,Any}("Milk",3),
                                                                    Pair{Any,Any}("Coffee",5),
                                                                    Pair{Any,Any}("Ukranian",2),
                                                                    Pair{Any,Any}("Japanese",5),
                                                                    Pair{Any,Any}("Snails",3),
                                                                    Pair{Any,Any}("Spaniard",4),
                                                                    Pair{Any,Any}("Water",1),
                                                                    Pair{Any,Any}("Winston",3),
                                                                    Pair{Any,Any}("Norwegian",1),
                                                                    Pair{Any,Any}("Fox",1),
                                                                    Pair{Any,Any}("Dog",4),
                                                                    Pair{Any,Any}("Ivory",4),
                                                                    Pair{Any,Any}("Englishman",3),
                                                                    Pair{Any,Any}("Yellow",1),
                                                                    Pair{Any,Any}("LuckyStrike",4),
                                                                    Pair{Any,Any}("Parliaments",5),
                                                                    Pair{Any,Any}("Blue",2),
                                                                    Pair{Any,Any}("Chesterfields",2)));
