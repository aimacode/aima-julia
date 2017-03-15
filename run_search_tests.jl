include("search.jl");

using Base.Test;

#The following search tests are from the aima-python doctest

@test depth_first_tree_search(NQueensProblem(8)) == Node{Array{Int64, 1}}([8, 4, 1, 3, 6, 2, 7, 5]);