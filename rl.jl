
export PassiveADPAgentMDP, PassiveADPAgentProgram,
        PassiveTDAgentProgram, QLearningAgentProgram;

#=

	PassiveADPAgentMDP is a MDP implementation of AbstractMarkovDecisionProcess

    that consists of a MarkovDecisionProcess 'mdp'.

=#
struct PassiveADPAgentMDP{T} <: AbstractMarkovDecisionProcess
	mdp::MarkovDecisionProcess{T}


    function PassiveADPAgentMDP{T}(initial::T, actions_list::Set{T}, terminal_states::Set{T}, gamma::Float64) where T
		return new(MarkovDecisionProcess(initial, actions_list, terminal_states, Dict(), gamma=gamma));
	end
end

PassiveADPAgentMDP(initial, actions_list::Set, terminal_states::Set, gamma::Float64) = PassiveADPAgentMDP{typeof(initial)}(initial, actions_list, terminal_states, gamma);

"""
    reward(mdp::PassiveADPAgentMDP, state)

Return a reward based on the given 'state'.
"""
function reward(mdp::PassiveADPAgentMDP, state)
    return mdp.mdp.reward[state];
end

"""
    transition_model(mdp::PassiveADPAgentMDP, state, action)

Return a list of (P(s'|s, a), s') pairs given the state 's' and action 'a'.
"""
function transition_model(mdp::PassiveADPAgentMDP, state, action)
    return collect((v, k) for (k, v) in get!(mdp.mdp.transitions, (state, action), Dict()));
end

"""
    actions(mdp::PassiveADPAgentMDP, state)

Return a set of actions that are possible in the given state.
"""
function actions(mdp::PassiveADPAgentMDP, state)
    if (state in mdp.mdp.terminal_states)
        return Set{Nothing}([nothing]);
    else
        return mdp.mdp.actions;
    end
end

"""
    policy_evaluation(pi::Dict, U::Dict, mdp::PassiveADPAgentMDP; k::Int64=20)

Return the updated utilities of the MDP's states by applying the modified policy iteration
algorithm on the given Markov decision process 'mdp', utility function 'U', policy 'pi',
and number of Bellman updates to use 'k'.
"""
function policy_evaluation(pi::Dict, U::Dict, mdp::PassiveADPAgentMDP; k::Int64=20)
    for i in 1:k
        for state in mdp.mdp.states
            if (length(transition_model(mdp, state, pi[state])) != 0)
                U[state] = (reward(mdp, state)
                            + (mdp.mdp.gamma
                            * sum((p * U[state_prime] for (p, state_prime) in transition_model(mdp, state, pi[state])))));
            else
                U[state] = (reward(mdp, state) + (mdp.mdp.gamma * 0));
            end
        end
    end
    return U;
end

#=

    PassiveADPAgentProgram is a passive reinforcement learning agent based on

    adaptive dynamic programming (Fig. 21.2).

=#
mutable struct PassiveADPAgentProgram <: AgentProgram
    state   # can be any DataType, but check for Nothing DataType later
    action  # can be any DataType, but check for Nothing DataType later
    U::Dict
    pi::Dict
    mdp::PassiveADPAgentMDP
    N_sa::Dict
    N_s_prime_sa::Dict

    function PassiveADPAgentProgram(pi::Dict, mdp::T) where {T <: AbstractMarkovDecisionProcess}
        return new(nothing,
                    nothing,
                    Dict(),
                    pi,
                    PassiveADPAgentMDP(mdp.initial, mdp.actions, mdp.terminal_states, mdp.gamma),
                    Dict(),
                    Dict());
    end
end

function execute(padpap::PassiveADPAgentProgram, percept::Tuple{Any, Any})
    local r_prime::Float64;
    s_prime, r_prime = percept;

    push!(padpap.mdp.mdp.states, s_prime);
    if (!haskey(padpap.mdp.mdp.reward, s_prime))
        padpap.U[s_prime] = r_prime;
        padpap.mdp.mdp.reward[s_prime] = r_prime;
    end
    if (!(padpap.state === nothing))
        padpap.N_sa[(padpap.state, padpap.action)] = get!(padpap.N_sa, (padpap.state, padpap.action), 0) + 1;
        padpap.N_s_prime_sa[(s_prime, padpap.state, padpap.action)] = get!(padpap.N_s_prime_sa, (s_prime, padpap.state, padpap.action), 0) + 1;
        for t in collect(result_state
                        for ((result_state, state, action), occurrences) in padpap.N_s_prime_sa
                        if (((state, action) == (padpap.state, padpap.action)) && (occurrences != 0)))
            get!(padpap.mdp.mdp.transitions, (padpap.state, padpap.action), Dict())[t] = padpap.N_s_prime_sa[(t, padpap.state, padpap.action)] / padpap.N_sa[(padpap.state, padpap.action)];
        end
    end
    local U::Dict = policy_evaluation(padpap.pi, padpap.U, padpap.mdp);
    if (s_prime in padpap.mdp.mdp.terminal_states)
        padpap.state = nothing;
        padpap.action = nothing;
    else
        padpap.state = s_prime;
        padpap.action = padpap.pi[s_prime];
    end
    return padpap.action;
end

function update_state(padpap::PassiveADPAgentProgram, percept::Tuple{Any, Any})
    return percept;
end

#=

    PassiveTDAgentProgram is a passive reinforcement learning agent that learns

    utility estimates by using temporal differences (Fig. 21.4).

=#
mutable struct PassiveTDAgentProgram <: AgentProgram
    state   # can be any DataType, but check for Nothing DataType later
    action  # can be any DataType, but check for Nothing DataType later
    reward  # can be any DataType, but check for Nothing DataType later
    gamma::Float64
    U::Dict
    pi::Dict
    N_s::Dict
    terminal_states::Set
    alpha::Function

    function PassiveTDAgentProgram(pi::Dict, mdp::T; alpha::Union{Nothing, Function}=nothing) where {T <: AbstractMarkovDecisionProcess}
        local gamma::Float64;
        local terminal_states::Set;
        local new_alpha::Function;
        if (typeof(mdp) <: PassiveADPAgentMDP)
            gamma = mdp.mdp.gamma;
            terminal_states = mdp.mdp.terminal_states;
        else
            gamma = mdp.gamma;
            terminal_states = mdp.terminal_states;
        end
        if (typeof(alpha) <: Nothing)
            new_alpha = (function(n::Number)
                            return (1/(n + 1));
                        end);
        else
            new_alpha = alpha;
        end
        return new(nothing,
                    nothing,
                    nothing,
                    gamma,
                    Dict(),
                    pi,
                    Dict(),
                    terminal_states,
                    new_alpha);
    end
end

function execute(ptdap::PassiveTDAgentProgram, percept::Tuple{Any, Any})
    local r_prime::Float64;
    s_prime, r_prime = update_state(ptdap, percept);
    if (!haskey(ptdap.N_s, s_prime))
        ptdap.U[s_prime] = r_prime;
    end
    if (!(ptdap.state === nothing))
        ptdap.N_s[ptdap.state] = get!(ptdap.N_s, ptdap.state, 0) + 1;
        ptdap.U[ptdap.state] = (get!(ptdap.U, ptdap.state, 0.0)
                                    + ptdap.alpha(get!(ptdap.N_s, ptdap.state, 0))
                                    * (ptdap.reward
                                    + (ptdap.gamma * get!(ptdap.U, s_prime, 0.0))
                                    - get!(ptdap.U, ptdap.state, 0.0)));
    end
    if (s_prime in ptdap.terminal_states)
        ptdap.state = nothing;
        ptdap.action = nothing;
        ptdap.reward = nothing;
    else
        ptdap.state = s_prime;
        ptdap.action = ptdap.pi[s_prime];
        ptdap.reward = r_prime;
    end
    return ptdap.action;
end

function update_state(ptdap::PassiveTDAgentProgram, percept::Tuple{Any, Any})
    return percept;
end

#=

    QLearningAgentProgram is an exploratory Q-learning agent that learns the value

    Q(state, action) for each action in each situation (Fig. 21.8). The agent uses the

    same exploration function as the exploratory ADP agent, but avoid learning the

    transition model because the Q-value of a state can be related directly to those of

    its neighbor.

=#
mutable struct QLearningAgentProgram <: AgentProgram
    state   # can be any DataType, but check for Nothing DataType later
    action  # can be any DataType, but check for Nothing DataType later
    reward  # can be any DataType, but check for Nothing DataType later
    gamma::Float64
    Q::Dict
    N_sa::Dict
    actions::Set
    terminal_states::Set
    R_plus::Float64 # optimistic estimate of the best possible reward obtainable
    N_e::Int64  # try action-state pair at least N_e times
    f::Function
    alpha::Function

    function QLearningAgentProgram(mdp::T, N_e::Int64, R_plus::Number; alpha::Union{Nothing, Function}=nothing) where {T <: AbstractMarkovDecisionProcess}
        local new_alpha::Function;
        local gamma::Float64;
        local actions::Set;
        local terminal_states::Set;
        if (typeof(mdp) <: PassiveADPAgentMDP)
            gamma = mdp.mdp.gamma;
            actions = mdp.mdp.actions;
            terminal_states = mdp.mdp.terminal_states;
        else
            gamma = mdp.gamma;
            actions = mdp.actions;
            terminal_states = mdp.terminal_states;
        end
        if (typeof(alpha) <: Nothing)
            new_alpha = (function(n::Number)
                            return (1/(n + 1));
                        end);
        else
            new_alpha = alpha;
        end
        return new(nothing,
                    nothing,
                    nothing,
                    gamma,
                    Dict(),
                    Dict(),
                    actions,
                    terminal_states,
                    R_plus,
                    N_e,
                    exploration_function,
                    new_alpha);
    end
end

function exploration_function(qlap::QLearningAgentProgram, u::Number, n::Number)
    if (n < qlap.N_e)  
        return qlap.R_plus;
    else
        return u;
    end
end

function actions(qlap::QLearningAgentProgram, state)
    if (state in qlap.terminal_states)
        return Set([nothing]);
    else
        return qlap.actions;
    end
end

function execute(qlap::QLearningAgentProgram, percept::Tuple{Any, Any})
    local r_prime::Float64;
    s_prime, r_prime = update_state(qlap, percept);
    if (!(qlap.state === nothing))
        if (qlap.state in qlap.terminal_states)
            qlap.Q[(qlap.state, nothing)] = r_prime;
        end
        qlap.N_sa[(qlap.state, qlap.action)] = get!(qlap.N_sa, (qlap.state, qlap.action), 0) + 1;
        # Default value for Q keys is 0.0.
        get!(qlap.Q, (qlap.state, qlap.action), 0.0);
        qlap.Q[(qlap.state, qlap.action)] = (qlap.Q[(qlap.state, qlap.action)]
                                                    + (qlap.alpha(qlap.N_sa[(qlap.state, qlap.action)]) *
                                                    (qlap.reward +
                                                    (qlap.gamma * reduce(max, collect(get!(qlap.Q, (s_prime, a_prime), 0.0)
                                                                                    for a_prime in actions(qlap, s_prime))))
                                                    - qlap.Q[(qlap.state, qlap.action)])));
    end
    if (!(qlap.state === nothing) && (qlap.state in qlap.terminal_states))
        qlap.state = nothing;
        qlap.action = nothing;
        qlap.reward = nothing;
    else
        qlap.state = s_prime;
        qlap.action = argmax(collect(actions(qlap, s_prime)),
                            (function(a_prime)
                                return qlap.f(qlap, get!(qlap.Q, (s_prime, a_prime), 0.0), get!(qlap.N_sa, (s_prime, a_prime), 0));
                            end));
        qlap.reward = r_prime;
    end
    return qlap.action;
end

function update_state(qlap::QLearningAgentProgram, percept::Tuple{Any, Any})
    return percept;
end

"""
    take_single_action(mdp::T, state, action) where {T <: AbstractMarkovDecisionProcess}

Return the next state by choosing a weighted sample of the resulting states for
taking the action 'action' in state 'state'.
"""
function take_single_action(mdp::T, state, action) where {T <: AbstractMarkovDecisionProcess}
    local x::Float64 = rand(RandomDeviceInstance);
    local cumulative_probability::Float64 = 0.0;
    for (p, state_p) in transition_model(mdp, state, action)
        cumulative_probability = cumulative_probability + p;
        if (x < cumulative_probability)
            return state_p;
        end
    end
    error("take_single_action(): Could not find a valid resulting state for the state ", state,
        " and action ", action, "!");
end

"""
    run_single_trial(ap::T1, mdp::T2) where {T1 <: AgentProgram, T2 <: AbstractMarkovDecisionProcess}

The agent program 'ap' executes a trial in the environment represented by the MDP 'mdp'.
"""
function run_single_trial(ap::T1, mdp::T2) where {T1 <: AgentProgram, T2 <: AbstractMarkovDecisionProcess}
    current_state = mdp.initial;
    while (true)
        local current_reward::Float64 = reward(mdp, current_state);
        local percept::Tuple = (current_state, current_reward);
        next_action = execute(ap, percept);
        if (typeof(next_action) <: Nothing)
            break;
        end
        current_state = take_single_action(mdp, current_state, next_action);
    end
    return nothing;
end

