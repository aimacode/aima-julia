include("../aimajulia.jl");

using Base.Test;

using aimajulia;

#The following CSP tests are from the aima-python doctest

d = CSPDict(ConstantFunctionDict(42));

@test d["life"] == 42;

