include("../aimajulia.jl");

using Test;

using Main.aimajulia;

#The following search tests are from the aima-python doctest

@test depth_first_tree_search(NQueensProblem(8)) == Node{Array{Int64, 1}}([8, 4, 1, 3, 6, 2, 7, 5]);

#Specify wordlist path for travis-ci testing.
filename = "./aima-data/EN-text/wordlist.txt";

@test length(BoggleFinder(board=collect("SARTELNID"), fn=filename)) == 206;

ab = GraphProblem("A", "B", aimajulia.romania);

@test solution(breadth_first_tree_search(ab)) == ["S", "F", "B"];

@test solution(breadth_first_search(ab)) == ["S", "F", "B"];

@test solution(uniform_cost_search(ab)) == ["S", "R", "P", "B"];

@test solution(depth_first_graph_search(ab)) == ["T", "L", "M", "D", "C", "P", "B"];

@test solution(iterative_deepening_search(ab)) == ["S", "F", "B"];

@test length(solution(depth_limited_search(ab))) == 50;

@test solution(astar_search(ab)) == ["S", "R", "P", "B"];

@test solution(recursive_best_first_search(ab)) == ["S", "R", "P", "B"];

@test compare_searchers([GraphProblem("A", "B", aimajulia.romania),
                        GraphProblem("O", "N", aimajulia.romania),
                        GraphProblem("Q", "WA", aimajulia.australia)],
                        ["Searcher", "Romania(A, B)", "Romania(O, N)", "Australia"]) == 
    ["Searcher"                     "Romania(A, B)"         "Romania(O, N)"         "Australia";
    "Main.aimajulia.breadth_first_tree_search"     "<  23/  24/  63/B>"    "<1191/1192/3378/N>"    "<   9/  10/  32/WA>";
    "Main.aimajulia.breadth_first_search"          "<   7/  11/  18/B>"    "<  18/  20/  44/N>"    "<   3/   6/   9/WA>";
    "Main.aimajulia.depth_first_graph_search"      "<   8/   9/  20/B>"    "<  16/  17/  37/N>"    "<   2/   3/   8/WA>";
    "Main.aimajulia.iterative_deepening_search"    "<  13/  36/  36/B>"    "< 683/1874/1875/N>"    "<   4/  13/  12/WA>";
    "Main.aimajulia.depth_limited_search"          "<  64/  94/ 167/B>"    "< 948/2629/2701/N>"    "<  51/  57/ 153/WA>";
    "Main.aimajulia.recursive_best_first_search"   "<  11/  12/  35/B>"    "<8481/8482/23788/N>"   "<  10/  11/  38/WA>"];

# Initialize LRTAStarAgentProgram with an OnlineSearchProblem.

lrtastar_program = OnlineSearchProblem("State_3", "State_5", aimajulia.one_dim_state_space, aimajulia.one_dim_state_space_least_costs);
lrtastar_agentprogram = LRTAStarAgentProgram(lrtastar_program);

@test execute(lrtastar_agentprogram, "State_3") == "Right";

@test execute(lrtastar_agentprogram, "State_4") == "Left";

@test execute(lrtastar_agentprogram, "State_3") == "Right";

@test execute(lrtastar_agentprogram, "State_4") == "Right";

@test execute(lrtastar_agentprogram, "State_5") == nothing;

lrtastar_agentprogram = LRTAStarAgentProgram(lrtastar_program);

@test execute(lrtastar_agentprogram, "State_4") == "Left";

lrtastar_agentprogram = LRTAStarAgentProgram(lrtastar_program);

@test execute(lrtastar_agentprogram, "State_5") == nothing;

