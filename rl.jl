
export PassiveADPAgentMDP, PassiveADPAgentProgram;

#=

	PassiveADPAgentMDP is a MDP implementation of AbstractMarkovDecisionProcess

    that consists of a MarkovDecisionProcess 'mdp'.

=#
type PassiveADPAgentMDP{T} <: AbstractMarkovDecisionProcess
	mdp::MarkovDecisionProcess{T}


	function PassiveADPAgentMDP{T}(initial::T, actions_list::Set{T}, terminal_states::Set{T}, gamma::Float64, states::Set{T})
		return new(MarkovDecisionProcess(initial, actions_list, terminal_states, Dict(), states=states, gamma=gamma));
	end
end

PassiveADPAgentMDP(initial, actions_list::Set, terminal_states::Set, gamma::Float64, states::Set) = PassiveADPAgentMDP{typeof(initial)}(initial, actions_list, terminal_states, gamma, states);

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
    return collect((v, k) for (k, v) in mdp.mdp.transitions);
end

"""
    actions(mdp::PassiveADPAgentMDP, state)

Return a set of actions that are possible in the given state.
"""
function actions(mdp::PassiveADPAgentMDP, state)
    if (state in mdp.mdp.terminal_states)
        return Set{Void}([nothing]);
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
            U[state] = (reward(mdp, state)
                        + (mdp.mdp.gamma
                        * sum((p * U[state_prime] for (p, state_prime) in transition_model(mdp, state, pi[state])))));
        end
    end
    return U;
end

#=

    PassiveADPAgentProgram is a passive reinforcement learning agent based on

    adaptive dynamic programming (Fig. 21.2).

=#
type PassiveADPAgentProgram <: AgentProgram
    state::Nullable
    action::Nullable
    U::Dict
    pi::Dict
    mdp::PassiveADPAgentMDP
    N_sa::Dict
    N_s_prime_sa::Dict

    function PassiveADPAgentProgram{T <: AbstractMarkovDecisionProcess}(pi::Dict, mdp::T)
        return new(Nullable(),
                    Nullable(),
                    Dict(),
                    pi,
                    PassiveADPAgentMDP(mdp.initial, mdp.actions, mdp.terminal_states, mdp.gamma, mdp.states),
                    Dict(),
                    Dict());
    end
end

function execute(padpap::PassiveADPAgentProgram, percept::Tuple{Any, Any})
    local r_prime::Float64;
    s_prime, r_prime = percept;

    union!(padpap.mdp.mdp.states, s_prime);
    if (!haskey(mdp.mdp.reward, s_prime))
        padpap.U[s_prime] = r_prime;
        padpap.mdp.mdp.reward = r_prime;
    end
    if (!isnull(padpap.state))
        padpap.N_sa[(get(padpap.state), get(padpap.action))] = get!(padpap.N_sa, (get(padpap.state), get(padpap.action)), 0) + 1;
        padpap.N_s_prime_sa[(s_prime, get(padpap.state), get(padpap.action))] = get!(padpap.N_s_prime_sa, (s_prime, get(padpap.state), get(padpap.action)), 0) + 1;
        for t in collect(result_state
                        for ((result_state, state, action), occurrences) in padpap.N_s_prime_sa
                        if (((state, action) == (get(padpap.state), get(padpap.action))) && (occurrences != 0)))
            get!(padpap.mdp.mdp.transitions, (get(padpap.state), get(padpap.action)), Dict())[t] = padpap.N_s_prime_sa[(t, get(padpap.state), get(padpap.action))] / padpap.N_sa[(get(padpap.state), get(padpap.action))];
        end
    end
    local U::Dict = policy_evaluation(padpap.pi, padpap.U, padpap.mdp);
    if (s_prime in padpap.mdp.mdp.terminal_states)
        padpap.state = Nullable();
        padpap.action = Nullable();
    else
        padpap.state = Nullable(s_prime);
        padpap.action = Nullable(padpap.pi[s_prime]);
    end
    return padpap.action;
end

function update_state(padpap::PassiveADPAgentProgram, percept::Tuple{Any, Any})
    return percept;
end

