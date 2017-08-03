include("../aimajulia.jl");

using Base.Test;

using aimajulia;

#The following mdp tests are from the aima-python doctests

@test (repr(value_iteration(aimajulia.sequential_decision_environment, epsilon=0.01)) == 
        "Dict((2,3)=>0.486437,(2,1)=>0.398102,(3,1)=>0.509285,(1,4)=>0.129589,(3,3)=>0.795361,(1,3)=>0.344613,(3,2)=>0.649581,(2,4)=>-1.0,(1,1)=>0.295435,(1,2)=>0.253487,(3,4)=>1.0)");

