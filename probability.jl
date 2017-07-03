
import Base: getindex, setindex!, values;

export getindex, setindex!, values;


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
    # Select the action with highest expected utility based on based on outcome probabilities and utility values.
    local action = argmax(dtap.actions, dtap.compute_probabilities);
    return action;
end

function update_state(dtap::DecisionTheoreticAgentProgram, state::String, percept::Tuple{Any, Any})
    println("update_state() is not implemented yet for ", typeof(dtap), "!");
    nothing;
end

abstract AbstractProbabilityDistribution;

type ProbabilityDistribution <: AbstractProbabilityDistribution
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

function getindex(pd::ProbabilityDistribution, key)
    if (haskey(pd.probabilities, key))
        return pd.probabilities[key];
    else
        return 0;
    end
end

function setindex!(pd::ProbabilityDistribution, key, value)
    if (!(key in pd.values))
        push!(pd.values, key);
    end
    pd.probabilities[key] = value;
    nothing;
end

function normalize{T <: AbstractProbabilityDistribution}(pd::T; epsilon::Float64=1e-09)
    local total::Float64 = sum(values(pd.probabilities));
    if (!((1.0 - epsilon) < total < (1.0 + epsilon))
        for k in keys(pd.probabilities)
            pd.probabilities[k] = pd.probabilities[k] / total;
        end
    end
    return pd;
end

function show_approximation{T <: AbstractProbabilityDistribution}(pd::T; number_format::String="%.4g")
    return join(collect(@sprintf("%s: "*number_format, v, k) for (k, v) in pd.probabilities), ", ");
end

function event_values(event::Tuple, variables::AbstractVector)
    if (length(event) == length(variables))
        return event;
    else
        error("event_values(): Length of ", event, " does not match length of ", variables, "!");
    end
end

function event_values(event::Dict, variables::AbstractVector)
    return Tuple((collect(event[v] for v in variables)...));
end

type JointProbabilityDistribution <: AbstractProbabilityDistribution
    variables::AbstractVector
    probabilities::Dict
    values::Dict{Any, AbstractVector}

    function JointProbabilityDistribution(variables::AbstractVector)
        return new(variables, Dict(), Dict{Any, AbstractVector}());
    end
end

function getindex(jpd::JointProbabilityDistribution, key_values)
    local key::Tuple = event_values(key_values, jpd.variables);
    if (haskey(jpd.probabilities, key))
        return jpd.probabilities[key];
    else
        return 0;
    end
end

function setindex!(jpd::JointProbabilityDistribution, key_values, value)
    local key::Tuple = event_values(key_values, jpd.variables);
    jpd.probabilities[key] = value;
    for (k, v) in zip(jpd.variables, key)
        if (!(v in jpd.values[k]))
            push!(jpd.values[k], v);
        end
    end
    nothing;
end

function values(jpd::JointProbabilityDistribution, key)
    return jpd.values(key);
end

function enumerate_joint{T <: AbstractProbabilityDistribution}(variables::AbstractVector, e::Dict, P::T)
    if (length(variables) == 0)
        return P[e];
    else
        local Y, rest::AbstractVector = variables[1], variables[2:end];
        return sum(collect(enumerate_joint(rest, extend(e, Y, y), P) for y in P.values(Y)));
    end
end

function enumerate_joint_ask{T <: AbstractProbabilityDistribution}(X::String, e::Dict, P::T)
    if (X in e)
        error("enumerate_joint_ask(): The query variable was not distinct from evidence variables.");
    end
    local Q::ProbabilityDistribution = ProbabilityDistribution(variable_name=X);
    local Y::AbstractVector = collect(v for v in P.variables if ((v != X) && !(v in e)));
    for x_i in P.values(X)
        Q[x_i] = enumerate_joint(Y, extend(e, X, xi), P)
    end
    return normalize(Q);
end

