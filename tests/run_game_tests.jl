include("../aimajulia.jl");

using Base.Test;

using aimajulia;

#The following Game tests are from the aima-python doctest

@test minimax_decision("A", Figure52Game()) == "A1";

@test alphabeta_full_search("A", Figure52Game()) == "A1";

@test alphabeta_search("A", Figure52Game()) == "A1";
