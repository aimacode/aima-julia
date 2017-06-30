

#=

    DecisionTheoreticAgentProgram is a decision-theoretic agent (Fig. 13.1).

=#
type DecisionTheoreticAgentProgram <: AgentProgram
    state::String
    goal::Nullable{String}
    actions::AbstractVector
    compute_probabilities::Function     # compute outcome probabilities and utility of an action

    function DecisionTheoreticAgentProgram(;initial_state::Union{Void, String}=nothing)
        return new(initial_state, Nullable{String}(), []);
    end
end

function execute(dtap::DecisionTheoreticAgentProgram, percept)
    dtap.state = update_state(dtap, dtap.state, percept);
    # Select the action with highest expected utility based on based on outcome probabilties and utility values.
    local action = argmax(dtap.actions, dtap.compute_probabilities);
    return action;
end

function update_state(dtap::DecisionTheoreticAgentProgram, state::String, percept::Tuple{Any, Any})
    println("update_state() is not implemented yet for ", typeof(dtap), "!");
    nothing;
end

type ProbabilityDistribution
    variable_name::String
    probabilities::Dict
    values::Array{Float64, 1}

    function ProbabilityDistribution(;variable_name::String="?", frequencies::Union{Void, Dict}=nothing)
        local pd::ProbabilityDistribution = new(variable_name, Dict(), []);
        if (!(typeof(frequencies) <: Void))
            for (k, v) in frequencies
                if (!(v in pd.values))
                    push!(pd.values, Float64(v));
                end
                pd.probabilities[k] = Float64(v);
            end
            normalize(pd);
        end
        return pd;
    end
end

function normalize(pd::ProbabilityDistribution; epsilon::Float64=1e-09)
    local total::Float64 = sum(values(pd.probabilities));
    if (!((1.0 - epsilon) < total < (1.0 + epsilon))
        for k in keys(pd.probabilities)
            pd.probabilities[k] = pd.probabilities[k] / total;
        end
    end
    return pd;
end

function show_approximation(pd::ProbabiltiyDistribution; number_format::String="%.4g")
    return join(collect(@sprintf("%s: "*number_format, v, k) for (k, v) in pd.probabilities), ", ");
end

