include("../aimajulia.jl");

using Base.Test;

using aimajulia;

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
    "aimajulia.breadth_first_tree_search"     "<  23/  24/  63/B>"    "<1191/1192/3378/N>"    "<   9/  10/  32/WA>";
    "aimajulia.breadth_first_search"          "<   7/  11/  18/B>"    "<  18/  20/  44/N>"    "<   3/   6/   9/WA>";
    "aimajulia.depth_first_graph_search"      "<   8/   9/  20/B>"    "<  16/  17/  37/N>"    "<   2/   3/   8/WA>";
    "aimajulia.iterative_deepening_search"    "<  13/  36/  36/B>"    "< 683/1874/1875/N>"    "<   4/  13/  12/WA>";
    "aimajulia.depth_limited_search"          "<  64/  94/ 167/B>"    "< 948/2629/2701/N>"    "<  51/  57/ 153/WA>";
    "aimajulia.recursive_best_first_search"   "<  11/  12/  35/B>"    "<8481/8482/23788/N>"   "<  10/  11/  38/WA>"];

