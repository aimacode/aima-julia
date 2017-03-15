include("search.jl");

using Base.Test;

#The following search tests are from the aima-python doctest

@test depth_first_tree_search(NQueensProblem(8)) == Node{Array{Int64, 1}}([8, 4, 1, 3, 6, 2, 7, 5]);

@test length(BoggleFinder(board=collect("SARTELNID"))) == 206;

ab = GraphProblem("A", "B", romania);

@test solution(breadth_first_tree_search(ab)) == ["S", "F", "B"];

@test solution(breadth_first_search(ab)) == ["S", "F", "B"];

@test solution(uniform_cost_search(ab)) == ["S", "R", "P", "B"];

@test solution(depth_first_graph_search(ab)) == ["T", "L", "M", "D", "C", "P", "B"];

@test solution(iterative_deepening_search(ab)) == ["S", "F", "B"];

@test length(solution(depth_limited_search(ab))) == 50;

@test solution(astar_search(ab)) == ["S", "R", "P", "B"];

@test solution(recursive_best_first_search(ab)) == ["S", "R", "P", "B"];

