
export AbstractMarkovDecisionProcess, MarkovDecisionProcess,
        reward, transition_model, actions;

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
        return new(initial, new_states, actions_list, terminal_states, transitions, gamma, reward);
    end  
end

MarkovDecisionProcess(initial, actions_list::Set, terminal_states::Set, transitions::Dict; states::Union{Void, Set}=nothing, gamma::Float64=Float64(0.9)) = MarkovDecisionProcess{typeof(initial)}(initial, actions_list, terminal_states, transitions, states, gamma);

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

