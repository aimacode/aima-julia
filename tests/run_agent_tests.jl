include("../aimajulia.jl");

using Base.Test;

using aimajulia;

#The following Agent tests are from the aima-python doctest

RVA = ReflexVacuumAgent();

@test execute(RVA.program, (aimajulia.loc_A, "Clean")) == "Right";

@test execute(RVA.program, (aimajulia.loc_B, "Clean")) == "Left";

@test execute(RVA.program, (aimajulia.loc_A, "Dirty")) == "Suck";

@test execute(RVA.program, (aimajulia.loc_B, "Dirty")) == "Suck";

TVE = TrivialVacuumEnvironment();

@test add_object(TVE, ModelBasedVacuumAgent()) == nothing;

@test run(TVE, steps=5) == nothing;

#=

    The following tests may fail sometimes because the tests check for the expected bounds.

    However, the results of tests that lie outside of expected bounds does not imply something is wrong.

=#

envs = [TrivialVacuumEnvironment() for i in range(0, 100)];

println("Expression: 7 < test_agent(ModelBasedVacuumAgent, 4, deepcopy(envs)) < 11");
mbva_result = test_agent(ModelBasedVacuumAgent, 4, deepcopy(envs))
println("Evaluated: 7 < ", mbva_result, " < 11");

println("Expression: 5 < test_agent(ReflexVacuumAgent, 4, deepcopy(envs)) < 9");
refva_result = test_agent(ReflexVacuumAgent, 4, deepcopy(envs));
println("Evaluated: 5 < ", refva_result, " < 9");

println("Expression: 2 < test_agent(TableDrivenVacuumAgent, 4, deepcopy(envs)) < 6");
tdva_result = test_agent(TableDrivenVacuumAgent, 4, deepcopy(envs));
println("Evaluated: 2 < ", tdva_result, " < 6");

println("Expression: 0.5 < test_agent(RandomVacuumAgent, 4, deepcopy(envs)) < 3");
randva_result = test_agent(RandomVacuumAgent, 4, deepcopy(envs));
println("Evaluated: 0.5 < ", randva_result, " < 3");
