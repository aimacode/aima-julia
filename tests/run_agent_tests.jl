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

@test 7 < test_agent(ModelBasedVacuumAgent, 4, deepcopy(envs)) < 11;

@test 5 < test_agent(ReflexVacuumAgent, 4, deepcopy(envs)) < 9;

@test 2 < test_agent(TableDrivenVacuumAgent, 4, deepcopy(envs)) < 6;

@test 0.5 < test_agent(RandomVacuumAgent, 4, deepcopy(envs)) < 3;
