
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

