include("../aimajulia.jl");

using Test;

using Main.aimajulia;

using Main.aimajulia.utils;

#The following reinforcement learning tests are from the aima-python doctests

north, south, west, east = (1, 0), (-1, 0), (0, -1), (0, 1);

policy = Dict([Pair((1, 1), north),
                Pair((1, 2), west),
                Pair((1, 3), west),
                Pair((1, 4), west),
                Pair((2, 1), north),
                Pair((2, 3), north),
                Pair((2, 4), nothing),
                Pair((3, 1), east),
                Pair((3, 2), east),
                Pair((3, 3), east),
                Pair((3, 4), nothing)])

passive_adp_agent = PassiveADPAgentProgram(policy, aimajulia.sequential_decision_environment);
for i in 1:75
    aimajulia.run_single_trial(passive_adp_agent, aimajulia.sequential_decision_environment);
end

@test (passive_adp_agent.U[(1, 1)] > 0.15);
println("passive_adp_agent.U[(1, 1)] (expected ~0.3): ", passive_adp_agent.U[(1, 1)]);

@test (passive_adp_agent.U[(2, 1)] > 0.15);
println("passive_adp_agent.U[(2, 1)] (expected ~0.4): ", passive_adp_agent.U[(2, 1)]);

@test (passive_adp_agent.U[(1, 2)] > 0);
println("passive_adp_agent.U[(1, 2)] (expected ~0.2): ", passive_adp_agent.U[(1, 2)]);

passive_td_agent = PassiveTDAgentProgram(policy,
                                        aimajulia.sequential_decision_environment,
                                        alpha=(function(n::Number)
                                                    return (60/(59+n));
                                                end));

for i in 1:200
    aimajulia.run_single_trial(passive_td_agent, aimajulia.sequential_decision_environment);
end

@test (passive_td_agent.U[(1, 1)] > 0.15);
println("passive_td_agent.U[(1, 1)] (expected ~0.3): ", passive_td_agent.U[(1, 1)]);

@test (passive_td_agent.U[(2, 1)] > 0.15);
println("passive_td_agent.U[(2, 1)] (expected ~0.35): ", passive_td_agent.U[(2, 1)]);

@test (passive_td_agent.U[(1, 2)] > 0.13);
println("passive_td_agent.U[(1, 2)] (expected ~0.25): ", passive_td_agent.U[(1, 2)]);

qlearning_agent = QLearningAgentProgram(aimajulia.sequential_decision_environment,
                                        5,
                                        2,
                                        alpha=(function(n::Number)
                                                    return (60/(59 + n));
                                                end));

for i in 1:200
    aimajulia.run_single_trial(qlearning_agent, aimajulia.sequential_decision_environment);
end

@test (qlearning_agent.Q[((2, 1), (1, 0))]>= -0.5);
println("qlearning_agent.Q[((2, 1), (1, 0))] expected (0.1): ", qlearning_agent.Q[((2, 1), (1, 0))]);

@test (qlearning_agent.Q[((1, 2), (-1, 0))] <= 0.5);
println("qlearning_agent.Q[((1, 2), (-1, 0))] expected (-0.1): ", qlearning_agent.Q[((1, 2), (-1, 0))]);

