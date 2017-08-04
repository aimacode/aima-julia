
export AbstractMarkovDecisionProcess, MarkovDecisionProcess,
        reward, transition_model, actions,
        GridMarkovDecisionProcess, go_to, show_grid, to_arrows,
        value_iteration, expected_utility, optimal_policy,
        policy_evaluation, policy_iteration;

abstract AbstractMarkovDecisionProcess;

#=

    MarkovDecisionProcess is a MDP implementation of AbstractMarkovDecisionProcess.

    A Markov decision process is a sequential decision problem with fully observable

    and stochastic environment with a transition model and rewards function.

    The discount factor (gamma variable) describes the preference for current rewards

    over future rewards.

=#
type MarkovDecisionProcess{T} <: AbstractMarkovDecisionProcess
    initial::T
    states::Set{T}
    actions::Set{T}
    terminal_states::Set{T}
    transitions::Dict
    gamma::Float64
    reward::Dict

    function MarkovDecisionProcess{T}(initial::T, actions_list::Set{T}, terminal_states::Set{T}, transitions::Dict, states::Union{Void, Set{T}}, gamma::Float64)
        if (!(0 < gamma <= 1))
            error("MarkovDecisionProcess(): The gamma variable of an MDP must be between 0 and 1, the constructor was given ", gamma, "!");
        end
        local new_states::Set{typeof(initial)};
        if (typeof(states) <: Set)
            new_states = states;
        else
            new_states = Set{typeof(initial)}();
        end
        return new(initial, new_states, actions_list, terminal_states, transitions, gamma, Dict());
    end  
end

MarkovDecisionProcess(initial, actions_list::Set, terminal_states::Set, transitions::Dict; states::Union{Void, Set}=nothing, gamma::Float64=0.9) = MarkovDecisionProcess{typeof(initial)}(initial, actions_list, terminal_states, transitions, states, gamma);

"""
    reward{T <: AbstractMarkovDecisionProcess}(mdp::T, state)

Return a reward based on the given 'state'.
"""
function reward{T <: AbstractMarkovDecisionProcess}(mdp::T, state)
    return mdp.reward[state];
end

"""
    transition_model{T <: AbstractMarkovDecisionProcess}(mdp::T, state, action)

Return a list of (P(s'|s, a), s') pairs given the state 's' and action 'a'.
"""
function transition_model{T <: AbstractMarkovDecisionProcess}(mdp::T, state, action)
    if (length(mdp.transitions) == 0)
        error("transition_model(): The transition model for the given 'mdp' could not be found!");
    else
        return mdp.transitions[state][action];
    end
end

"""
    actions{T <: AbstractMarkovDecisionProcess}(mdp::T, state)

Return a set of actions that are possible in the given state.
"""
function actions{T <: AbstractMarkovDecisionProcess}(mdp::T, state)
    if (state in mdp.terminal_states)
        return Set{Void}([nothing]);
    else
        return mdp.actions;
    end
end

#=

    GridMarkovDecisionProcess is a two-dimensional environment MDP implementation

    of AbstractMarkovDecisionProcess. Obstacles in the environment are represented

    by a null.
=#
type GridMarkovDecisionProcess <: AbstractMarkovDecisionProcess
    initial::Tuple{Int64, Int64}
    states::Set{Tuple{Int64, Int64}}
    actions::Set{Tuple{Int64, Int64}}
    terminal_states::Set{Tuple{Int64, Int64}}
    grid::Array{Nullable{Float64}, 2}
    gamma::Float64
    reward::Dict

    function GridMarkovDecisionProcess(initial::Tuple{Int64, Int64}, terminal_states::Set{Tuple{Int64, Int64}}, grid::Array{Nullable{Float64}, 2}; states::Union{Void, Set{Tuple{Int64, Int64}}}=nothing, gamma::Float64=0.9)
        if (!(0 < gamma <= 1))
            error("GridMarkovDecisionProcess(): The gamma variable of an MDP must be between 0 and 1, the constructor was given ", gamma, "!");
        end
        local new_states::Set{Tuple{Int64, Int64}};
        if (typeof(states) <: Set)
            new_states = states;
        else
            new_states = Set{Tuple{Int64, Int64}}();
        end
        local orientations::Set = Set{Tuple{Int64, Int64}}([(1, 0), (0, 1), (-1, 0), (0, -1)]);
        # (0, 1) will move the agent east.
        local reward::Dict = Dict();
        for i in 1:getindex(size(grid), 1)
            for j in 1:getindex(size(grid, 2))
                reward[(i, j)] = grid[i, j]
                if (!isnull(grid[i, j]))
                    push!(new_states, (i, j));
                end
            end
        end
        return new(initial, new_states, orientations, terminal_states, grid, gamma, reward);
    end 
end

"""
    go_to(gmdp::GridMarkovDecisionProcess, state::Tuple{Int64, Int64}, direction::Tuple{Int64, Int64})

Return the next state given the current state and direction.
"""
function go_to(gmdp::GridMarkovDecisionProcess, state::Tuple{Int64, Int64}, direction::Tuple{Int64, Int64})
    local next_state::Tuple{Int64, Int64} = map(+, state, direction);
    if (next_state in gmdp.states)
        return next_state;
    else
        return state;
    end
end

function transition_model(gmdp::GridMarkovDecisionProcess, state::Tuple{Int64, Int64}, action::Void)
    return [(0.0, state)];
end

function transition_model(gmdp::GridMarkovDecisionProcess, state::Tuple{Int64, Int64}, action::Tuple{Int64, Int64})
    return [(0.8, go_to(gmdp, state, action)),
            (0.1, go_to(gmdp, state, utils.turn_heading(action, -1))),
            (0.1, go_to(gmdp, state, utils.turn_heading(action, 1)))];
end

function show_grid(gmdp::GridMarkovDecisionProcess, mapping::Dict)
    local grid::Array{Nullable{String}, 2};
    local rows::AbstractVector = [];
    for i in 1:getindex(size(gmdp.grid), 1)
        local row::Array{Nullable{String}, 1} = Array{Nullable{String}, 1}();
        for j in 1:getindex(size(gmdp.grid), 2)
            push!(row, Nullable{String}(get(mapping, (i, j), nothing)));
        end
        push!(rows, reshape(row, (1, length(row))));
    end
    grid = reduce(vcat, rows);
    return grid;
end

function to_arrows(gmdp::GridMarkovDecisionProcess, policy::Dict)
    local arrow_characters::Dict = Dict([Pair((0, 1), ">"),
                                        Pair((-1, 0), "^"),
                                        Pair((0, -1), "<"),
                                        Pair((1, 0), "v"),
                                        Pair(nothing, ".")]);
    return show_grid(gmdp, Dict(collect(Pair(state, arrow_characters[action])
                                    for (state, action) in policy)));
end

# An example sequential decision problem (Fig. 17.1a) where an agent does not
# terminate until it reaches a terminal state in the 4x3 environment (Fig. 17.1a).
#
# Matrices in Julia start from the upper-left corner and index (1, 1).
sequential_decision_environment = GridMarkovDecisionProcess((0, 0),
                                            Set([(2, 4), (3, 4)]),
                                            map(Nullable{Float64}, [-0.04 -0.04 -0.04 -0.04;
                                                                    -0.04 nothing -0.04 -1;
                                                                    -0.04 -0.04 -0.04 +1]));

"""
    value_iteration{T <: AbstractMarkovDecisionProcess}(mdp::T; epsilon::Float64=0.001)

Return the utilities of the MDP's states as a Dict by applying the value iteration algorithm (Fig. 17.4)
on the given Markov decision process 'mdp' and a arbitarily small positive number 'epsilon'.
"""
function value_iteration{T <: AbstractMarkovDecisionProcess}(mdp::T; epsilon::Float64=0.001)
    local U_prime::Dict = Dict(collect(Pair(state, 0.0) for state in mdp.states));
    while (true)
        local U::Dict = copy(U_prime);
        local delta::Float64 = 0.0;
        for state in mdp.states
            U_prime[state] = (reward(mdp, state)
                                + (mdp.gamma
                                * max((sum(collect(p * U[state_prime] 
                                                    for (p, state_prime) in transition_model(mdp, state, action)))
                                        for action in actions(mdp, state))...)));
            delta = max(delta, abs(U_prime[state] - U[state]));
        end
        if (delta < ((epsilon * (1 - mdp.gamma))/mdp.gamma))
            return U;
        end
    end
end

function value_iteration(gmdp::GridMarkovDecisionProcess; epsilon::Float64=0.001)
    local U_prime::Dict = Dict(collect(Pair(state, 0.0) for state in gmdp.states));
    while (true)
        local U::Dict = copy(U_prime);
        local delta::Float64 = 0.0;
        for state in gmdp.states
            # Extract Float64 from Nullable{Float64}
            U_prime[state] = (get(reward(gmdp, state))
                            + (gmdp.gamma 
                            * max((sum(collect(p * U[state_prime] 
                                                for (p, state_prime) in transition_model(gmdp, state, action)))
                                        for action in actions(gmdp, state))...)));
            delta = max(delta, abs(U_prime[state] - U[state]));
        end
        if (delta < ((epsilon * (1 - gmdp.gamma))/gmdp.gamma))
            return U;
        end
    end
end

#=
function expected_utility{T <: AbstractMarkovDecisionProcess}(mdp::T, U::Dict, state::Tuple{Int64, Int64}, action::Tuple{Int64, Int64})
    return sum((p * U[state_prime] for (p, state_prime) in transition_model(mdp, state, action)));
end
=#

function expected_utility(mdp::AbstractMarkovDecisionProcess, U::Dict, state::Tuple{Int64, Int64}, action::Tuple{Int64, Int64})
    return sum((p * U[state_prime] for (p, state_prime) in transition_model(mdp, state, action)));
end

function expected_utility(mdp::AbstractMarkovDecisionProcess, U::Dict, state::Tuple{Int64, Int64}, action::Void)
    return sum((p * U[state_prime] for (p, state_prime) in transition_model(mdp, state, action)));
end

"""
    optimal_policy{T <: AbstractMarkovDecisionProcess}(mdp::T, U::Dict)

Return the optimal_policy 'π*(s)' (Equation 17.4) given the Markov decision process 'mdp'
and the utility function 'U'.
""" 
function optimal_policy{T <: AbstractMarkovDecisionProcess}(mdp::T, U::Dict)
    local pi::Dict = Dict();
    for state in mdp.states
        pi[state] = argmax(collect(actions(mdp, state)), (function(action::Union{Void, Tuple{Int64, Int64}})
                                                                return expected_utility(mdp, U, state, action);
                                                            end));
    end
    return pi;
end

"""
    policy_evaluation{T <: AbstractMarkovDecisionProcess}(pi::Dict, U::Dict, mdp::T; k::Int64=20)

Return the updated utilities of the MDP's states by applying the modified policy iteration
algorithm on the given Markov decision process 'mdp', utility function 'U', policy 'pi',
and number of Bellman updates to use 'k'.
"""
function policy_evaluation{T <: AbstractMarkovDecisionProcess}(pi::Dict, U::Dict, mdp::T; k::Int64=20)
    for i in 1:k
        for state in mdp.states
            U[state] = (reward(mdp, state)
                        + (mdp.gamma
                        * sum((p * U[state_prime] for (p, state_prime) in transition_model(mdp, state, pi[state])))));
        end
    end
    return U;
end

function policy_evaluation(pi::Dict, U::Dict, mdp::GridMarkovDecisionProcess; k::Int64=20)
    for i in 1:k
        for state in mdp.states
            U[state] = (get(reward(mdp, state))
                        + (mdp.gamma
                        * sum((p * U[state_prime] for (p, state_prime) in transition_model(mdp, state, pi[state])))));
        end
    end
    return U;
end

"""
    policy_iteration{T <: AbstractMarkovDecisionProcess}(mdp::T)

Return a policy using the policy iteration algorithm (Fig. 17.7) given the Markov decision process 'mdp'.
"""
function policy_iteration{T <: AbstractMarkovDecisionProcess}(mdp::T)
    local U::Dict = Dict(collect(Pair(state, 0.0) for state in mdp.states));
    local pi::Dict = Dict(collect(Pair(state, rand(RandomDeviceInstance, collect(actions(mdp, state))))
                                    for state in mdp.states));
    while (true)
        U = policy_evaluation(pi, U, mdp);
        local unchanged::Bool = true;
        for state in mdp.states
            local action = argmax(collect(actions(mdp, state)), (function(action::Union{Void, Tuple{Int64, Int64}})
                                                                    return expected_utility(mdp, U, state, action);
                                                                end));
            if (action != pi[state])
                pi[state] = action;
                unchanged = false;
            end
        end
        if (unchanged)
            return pi;
        end
    end
end

