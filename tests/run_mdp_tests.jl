include("../aimajulia.jl");

using Base.Test;

using aimajulia;

#The following mdp tests are from the aima-python doctests

tm = Dict([Pair("A", Dict([Pair("a1", (0.3, "B")), Pair("a2", (0.7, "C"))])),
            Pair("B", Dict([Pair("a1", (0.5, "B")), Pair("a2", (0.5, "A"))])),
            Pair("C", Dict([Pair("a1", (0.9, "A")), Pair("a2", (0.1, "B"))]))]);

mdp = MarkovDecisionProcess("A", Set(["a1", "a2"]), Set(["C"]), tm, states=Set(["A","B","C"]));

@test (transition_model(mdp, "A", "a1") == (0.3, "B"));

@test (transition_model(mdp, "B", "a2") == (0.5, "A"));

@test (transition_model(mdp, "C", "a1") == (0.9, "A"));

@test (repr(value_iteration(aimajulia.sequential_decision_environment, epsilon=0.01)) == 
        "Dict((2, 3)=>0.486437,(2, 1)=>0.398102,(3, 1)=>0.509285,(1, 4)=>0.129589,(3, 3)=>0.795361,(1, 3)=>0.344613,(3, 2)=>0.649581,(2, 4)=>-1.0,(1, 1)=>0.295435,(1, 2)=>0.253487,(3, 4)=>1.0)");

pi = optimal_policy(aimajulia.sequential_decision_environment, value_iteration(aimajulia.sequential_decision_environment, epsilon=0.01));

@test (repr(pi) == "Dict{Any,Any}(Pair{Any,Any}((2, 3), (1, 0)),Pair{Any,Any}((2, 1), (1, 0)),Pair{Any,Any}((3, 1), (0, 1)),Pair{Any,Any}((1, 4), (0, -1)),Pair{Any,Any}((3, 3), (0, 1)),Pair{Any,Any}((1, 3), (1, 0)),Pair{Any,Any}((3, 2), (0, 1)),Pair{Any,Any}((2, 4), nothing),Pair{Any,Any}((1, 1), (1, 0)),Pair{Any,Any}((1, 2), (0, 1)),Pair{Any,Any}((3, 4), nothing))");

@test (repr(to_arrows(aimajulia.sequential_decision_environment, pi)) == "Nullable{String}[\"v\" \">\" \"v\" \"<\"; \"v\" #NULL \"v\" \".\"; \">\" \">\" \">\" \".\"]");

@test (policy_iteration(aimajulia.sequential_decision_environment) == 
        Dict([Pair((2,3),(1,0)),
            Pair((2,1),(1,0)),
            Pair((3,1),(0,1)),
            Pair((1,4),(0,-1)),
            Pair((3,3),(0,1)),
            Pair((1,3),(1,0)),
            Pair((3,2),(0,1)),
            Pair((2,4),nothing),
            Pair((1,1),(1,0)),
            Pair((1,2),(0,1)),
            Pair((3,4),nothing)]));

