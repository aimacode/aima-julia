
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
    if (!((1.0 - epsilon) < total < (1.0 + epsilon)))
        for k in keys(pd.probabilities)
            pd.probabilities[k] = pd.probabilities[k] / total;
        end
    end
    return pd;
end

"""
    show_approximation(pd)

Return a String representing the sorted approximate values of the probabilities.

Note: The @sprintf macro does not allow string concatenation for its format string.
The usage of the macro requires the format string to be a static string.
"""
function show_approximation{T <: AbstractProbabilityDistribution}(pd::T)
    return join(collect(@sprintf("%s: %.4g", v, k) for (k, v) in sort(pd.probabilities)), ", ");
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
    return jpd.values[key];
end

function enumerate_joint{T <: AbstractProbabilityDistribution}(variables::AbstractVector, e::Dict, P::T)
    if (length(variables) == 0)
        return P[e];
    else
        local Y = variables[1];
        local rest::AbstractVector = variables[2:end];
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

type BayesianNetworkNode
    variable::String
    parents::Array{String, 1}
    cpt::Dict
    children::AbstractVector

    function BayesianNetworkNode{T <: Real}(X::String, parents::String, conditional_probability_table::T)
        local parents_array::Array{String, 1} = map(String, split(parents));
        local cpt::Dict = Dict([Pair((), conditional_probability_table)]);
        for (keys, value) in cpt
            if (!((typeof(keys) <: Tuple) & (length(keys) == length(parents_array))))
                error("BayesianNetworkNode(): The length of ", keys, " and ", parents, " do not match!");
            end
            if (!(all(typeof(key) <: Bool for key in keys)))
                error("BayesianNetworkNode(): ", keys, " should only have boolean values!");
            end
            if (!(0.0 <= value <= 1.0))
                error("BayesianNetworkNode(): The given value ", value, " is not a valid probability!");
            end
        end
        return new(X, parents_array, cpt, []);;
    end

    function BayesianNetworkNode(X::String, parents::String, conditional_probability_table::Dict)
        local parents_array::Array{String, 1} = map(String, split(parents));
        local cpt::Dict;
        if ((length(conditional_probability_table) != 0) & (typeof(first(keys(conditional_probability_table))) <: Bool))
            cpt = Dict(collect(Pair((value,), p) for (value, p) in conditional_probability_table));
        else
            cpt = conditional_probability_table;
        end
        for (keys, value) in cpt
            if (!((typeof(keys) <: Tuple) && (length(keys) == length(parents_array))))
                error("BayesianNetworkNode(): The length of ", keys, " and ", parents, " do not match!");
            end
            if (!(all(typeof(key) <: Bool for key in keys)))
                error("BayesianNetworkNode(): ", keys, " should only have boolean values!");
            end
            if (!(0.0 <= value <= 1.0))
                error("BayesianNetworkNode(): The given value ", value, " is not a valid probability!");
            end
        end
        return new(X, parents_array, cpt, []);;
    end

    function BayesianNetworkNode{T <: Real}(X::String, parents::Array{String, 1}, conditional_probability_table::T)
        local cpt::Dict = Dict([Pair((), conditional_probability_table)]);
        for (keys, value) in cpt
            if (!((typeof(keys) <: Tuple) & (length(keys) == length(parents))))
                error("BayesianNetworkNode(): The length of ", keys, " and ", parents, " do not match!");
            end
            if (!(all(typeof(key) <: Bool for key in keys)))
                error("BayesianNetworkNode(): ", keys, " should only have boolean values!");
            end
            if (!(0.0 <= value <= 1.0))
                error("BayesianNetworkNode(): The given value ", value, " is not a valid probability!");
            end
        end
        return new(X, parents, cpt, []);;
    end

    function BayesianNetworkNode(X::String, parents::Array{String, 1}, conditional_probability_table::Dict)
        local cpt::Dict;
        if ((length(conditional_probability_table) != 0) & (typeof(first(keys(conditional_probability_table))) <: Bool))
            cpt = Dict(collect(Pair((value,), p) for (value, p) in conditional_probability_table));
        else
            cpt = conditional_probability_table;
        end
        for (keys, value) in cpt
            if (!((typeof(keys) <: Tuple) && (length(keys) == length(parents))))
                error("BayesianNetworkNode(): The length of ", keys, " and ", parents, " do not match!");
            end
            if (!(all(typeof(key) <: Bool for key in keys)))
                error("BayesianNetworkNode(): ", keys, " should only have boolean values!");
            end
            if (!(0.0 <= value <= 1.0))
                error("BayesianNetworkNode(): The given value ", value, " is not a valid probability!");
            end
        end
        return new(X, parents, cpt, []);;
    end
end

function probability(bnn::BayesianNetworkNode, value::Bool, event::Dict)
    local probability_true::Float64 = bnn.cpt[event_values(event, bnn.parents)];
    if (value)
        return probability_true;
    else
        return 1.0 - probability_true;
    end
end

function sample(bnn::BayesianNetworkNode, event::Dict)
    return rand(RandomDeviceInstance) < probability(bnn, true, event);
end

#=

    BayesianNetwork is a Bayesian network that contains only boolean variable nodes.

=#
type BayesianNetwork
    variables::AbstractVector
    nodes::Array{BayesianNetworkNode, 1}

    function BayesianNetwork()
        return new([], Array{BayesianNetworkNode, 1}());
    end

    function BayesianNetwork(node_specifications::AbstractVector)
        local bn::BayesianNetwork = new([], Array{BayesianNetworkNode, 1}());
        for node_specification in node_specifications
            add_node(bn, node_specification);
        end
        return bn;
    end
end

function add_node(bn::BayesianNetwork, ns::Tuple)
    local node::BayesianNetworkNode = BayesianNetworkNode(ns...);
    if (node.variable in bn.variables)
        error("add_node(): The node's variable '", node.variable, "'' can't be used, as it already exists in the Bayesian network's variables!");
    end
    if (!all((parent in bn.variables) for parent in node.parents))
        error("add_node(): Detected a parent node that doesn't exist in the Bayesian network!");
    end
    push!(bn.nodes, node);
    push!(bn.variables, node.variable);
    for parent in node.parents
        local var_node::BayesianNetworkNode = variable_node(bn, parent);
        push!(var_node.children, node);
    end
    nothing;
end

function variable_node(bn::BayesianNetwork, v::String)
    for node in bn.nodes
        if (node.variable == v)
            return node;
        end
    end
    error("variable_node(): Could not find node with variable '", v, "'!");
end

"""
    variable_values(bn, variable)

Return the domain (possible variable values) of 'variable'.
"""
function variable_values(bn::BayesianNetwork, variable::String)
    return (true, false);
end

# A BayesianNetwork representing the burglary network example (Fig. 14.2)

burglary_network = BayesianNetwork([("Burglary", "", 0.001),
                                    ("Earthquake", "", 0.002),
                                    ("Alarm", "Burglary Earthquake", Dict([Pair((true, true), 0.95),
                                                                            Pair((true, false), 0.94),
                                                                            Pair((false, true), 0.29),
                                                                            Pair((false, false), 0.001)])),
                                    ("JohnCalls", "Alarm", Dict([Pair(true, 0.90), Pair(false, 0.05)])),
                                    ("MaryCalls", "Alarm", Dict([Pair(true, 0.70), Pair(false, 0.01)]))]);

function enumerate_all(variables::AbstractVector, event::Dict, bn::BayesianNetwork)
    if (length(variables) == 0)
        return 1.0;
    end
    local Y::String = variables[1];
    local rest::Array{String, 1} = variables[2:end];
    local Y_node::BayesianNetworkNode = variable_node(bn, Y);
    if (haskey(event, Y))
        return (probability(Y_node, event[Y], event) * enumerate_all(rest, event, bn));
    else
        return sum(probability(Y_node, y, event) * enumerate_all(rest, extend(event, Y, y), bn)
                    for y in variable_values(bn, Y));
    end
end

"""
    enumeration_ask(X::String, e::Dict, bn::BayesianNetwork)

Return the conditional probability distribution by applying the enumeration algorithm (Fig. 14.9)
to the given variable 'X', observed event 'e', and Bayesian network 'bn'.
"""
function enumeration_ask(X::String, e::Dict, bn::BayesianNetwork)
    if (haskey(e, X))
        error("enumeration_ask(): The query variable was not distinct from evidence variables.");
    end
    local Q::ProbabilityDistribution = ProbabilityDistribution(variable_name=X);
    for x_i in variable_values(bn, X)
        Q[x_i] = enumerate_all(bn.variables, extend(e, X, x_i), bn);
    end
    return normalize(Q);
end

function sum_out(key::String, factors::AbstractVector, bn::BayesianNetwork)
    local result::AbstractVector = [];
    local variable_factors::AbstractVector = [];
    for f in factors
        if (key in f.variables)
            push!(variable_factors, f);
        else
            push!(result, f);
        end
    end
    push!(result, sum_out(pointwise_product(variable_factors, bn), key, bn));
    return result;
end

"""
    elimination_ask(X::String, e::Dict, bn::BayesianNetwork)

Return the conditional probability distribution by applying the variable elimination algorithm (Fig. 14.11)
to the given variable 'X', observed event 'e', and Bayesian network 'bn'.
"""
function elimination_ask(X::String, e::Dict, bn::BayesianNetwork)
    if (haskey(e, X))
        error("enumeration_ask(): The query variable was not distinct from evidence variables.");
    end
    local factors::AbstractVector = [];
    for key in reverse(bn.variables)
        push!(factors, make_factor(key, e, bn));
        if (is_hidden(key, X, event))
            factors = sum_out(key, factors, bn);
        end
    end
    return normalize(pointwise_product(factors, bn));
end

function all_events(variables::AbstractVector, bn::BayesianNetwork, event::Dict)
    if (length(variables) == 0)
        return (event,);
    else
        local X::String = variables[1];
        local rest::AbstractVector = variables[2:end];
        local solution::Tuple = ();
        for e_1 in all_events(rest, bn, e)
            for x in variable_values(bn, X)
                solution = Tuple((solution..., extend(e_1, X, x)));
            end
        end
        return solution;
    end
end

# Factors are used in variable elimination when evaluating expression
# representations of Bayesian networks.

type Factor
    variables::AbstractVector
    cpt::Dict

    function Factor(variables::AbstractVector, cpt::Dict)
        return new(variables, cpt);
    end
end

function pointwise_product(f::Factor, other::Factor, bn::BayesianNetwork)
    local variables::AbstractVector = collect(union(Set(f.variables), Set(other.variables)));
    local cpt::Dict = Dict(collect(Pair(event_values(e, variables), probability(f, e) * probability(other, e))
                                    for e in all_events(variables, bn, Dict())));
    return Factor(variables, cpt);
end

function sum_out(f::Factor, key::String, bn::BayesianNetwork)
    local variables::AbstractVector = collect(X for x in f.variables if (X != key));
    local cpt::Dict = Dict(collect(Pair(event_values(e, variables), sum(probability(f, extend(e, key, value))
                                                                        for value in variable_values(bn, key)))
                                    for e in all_events(variables, bn, Dict())));
    return Factor(variables, cpt);
end

function normalize(f::Factor)
    if (length(f.variables) != 1)
        error("normalize(): The variables of factor", f, " must be length of 1!");
    end
    return ProbabilityDistribution(variable_name=f.variables[1],
                                    frequencies=Dict(collect(Pair(k, v)
                                                            for ((k,), v) in f.cpt)));
end

function probability(f::Factor, e::Dict)
    return f.cpt[event_values(e, f.variables)];
end

function is_hidden(key::String, X::String, event::Dict)
    return ((key != X) & (!haskey(event, key)));
end

function make_factor(key::String, event::Dict, bn::BayesianNetwork)
    local node::BayesianNetworkNode = variable_node(bn, var);
    local variables::AbstractVector = collect(X for X in vcat([key], node.parents) if (!haskey(event, X)));
    local cpt::Dict = Dict(collect(Pair(event_values(e_1, variables), probability(node, e_1[key], e_1))
                                    for e_1 in all_events(variables, bn, event)));
    return Factor(variables, cpt);
end

function pointwise_product(factors::AbstractVector, bn::BayesianNetwork)
    return reduce((function(f_1::Factor, f_2::Factor)
                        return pointwise_product(f_1, f_2, bn);
                    end), factors);
end

# A BayesianNetwork representing the sprinkler network example (Fig. 14.12a)

sprinkler_network = BayesianNetwork([("Cloudy", "", 0.5),
                                    ("Sprinkler", "Cloudy", Dict([Pair(true, 0.10), Pair(false, 0.50)])),
                                    ("Rain", "Cloudy", Dict([Pair(true, 0.80), Pair(false, 0.20)])),
                                    ("WetGrass", "Sprinkler Rain", Dict([Pair((true, true), 0.99),
                                                                        Pair((true, false), 0.90),
                                                                        Pair((false, true), 0.90),
                                                                        Pair((false, false), 0.00)]))]);

"""
    prior_sample(bn::BayesianNetwork)

Return an event as a Dict of 'variable=>value' pairs generated by a sampling algorithm (Fig. 14.13).
"""
function prior_sample(bn::BayesianNetwork)
    local event::Dict = Dict();
    for node in bn.nodes
        event[node.variable] = sample(node, event);
    end
    return event;
end

function consistent_with(event::Dict, evidence::Dict)
    return all(get(evidence, k, v) == v for (k, v) in event);
end

"""
    rejection_sampling(X::String, e::Dict, bn::BayesianNetwork, N::Int64)

Return an estimate of the probability distribution of variable 'X' by using the 
rejection-sampling algorithm (Fig. 14.14) on the given observed event 'e', Bayesian
network 'bn', and total number of samples to generate 'N'.
"""
function rejection_sampling(X::String, e::Dict, bn::BayesianNetwork, N::Int64)
    if (N < 0)
        error("rejection_sampling(): ", N, " is not a valid number of samples!");
    end
    local counts::Dict = Dict(collect(Pair(x, 0) for x in variable_values(bn, X)));
    for j in 1:N
        local sample::Dict = prior_sample(bn);
        if (consistent_with(sample, e))
            counts[sample[X]] = counts[sample[X]] + 1;
        end
    end
    return ProbabilityDistribution(variable_name=X, frequencies=counts);
end

function weighted_sample(bn::BayesianNetwork, e::Dict)
    local w::Float64;
    local event::Dict = copy(e);
    for node in bn.nodes
        local X_i::String = node.variable;
        if (haskey(e, X_i))
            w = w * probability(node, e[X_i], event);
        else
            event[X_i] = sample(node, event);
        end
    end
    return event, w;
end

"""
    likelihood_weighting(X::String, e::Dict, bn::BayesianNetwork, N::Int64)

Return an estimate of the probability distribution of variable 'X' by using the
likelihood-weighting algorithm (Fig. 14.15) on the given observed event 'e',
Bayesian network 'bn', and the total number of samples to generate 'N'.
"""
function likelihood_weighting(X::String, e::Dict, bn::BayesianNetwork, N::Int64)
    if (N < 0)
        error("likelihood_weighting(): ", N, " is not a valid number of samples!");
    end
    local W::Dict = Dict(collect(Pair(x, 0.0) for x in variable_values(bn, X)));
    for j in 1:N
        local sample::Dict;
        local weight::Float64;
        sample, weight = weighted_sample(bn, e);
        W[sample[X]] = W[sample[X]] + weight;
    end
    return ProbabilityDistribution(variable_name=X, frequencies=W);
end

"""
    markov_blanket_sampling(X::String, e::Dict, bn::BayesianNetwork)

Return a sample from P(X | mb) where 'mb' is the Markov blanket of 'X'.

The Markov blanket of 'X' is composed of its parents, children, and
children's parents.
"""
function markov_blanket_sample(X::String, e::Dict, bn::BayesianNetwork)
    local X_node::BayesianNetworkNode = variable_node(bn, X);
    local Q::ProbabilityDistribution = ProbabilityDistribution(variable_name=X);
    for x_i in variable_values(bn, X)
        local e_i::Dict = extend(e, X, x_i);
        # Using Equation 14.12, we calculate x_i's Markov blanket
        Q[x_i] = probability(X_node, x_i, e) * prod(probability(Y_j, e_i[Y_j.variable], e_i)
                                                    for Y_j in X_node.children);
    end
    # Return a boolean variable's value because BayesianNetwork
    # contains only boolean variable nodes.
    return probability(normalize(Q)[true]);
end

"""
    gibbs_ask(X::String, e::Dict, bn::BayesianNetwork, N::Int64)

Return an estimate of the probability distribution of variable 'X' by using the
Gibbs sampling algorithm (Fig. 14.16) on the given observed event 'e',
Bayesian network 'bn', and the total number of samples to generate 'N'.
"""
function gibbs_ask(X::String, e::Dict, bn::BayesianNetwork, N::Int64)
    if (N < 0)
        error("likelihood_weighting(): ", N, " is not a valid number of samples!");
    end
    if (haskey(e, X))
        error("gibbs_ask(): The query variable was not distinct from evidence variables.");
    end
    local counts::Dict = Dict(collect(Pair(x, 0.0) for x in variable_values(bn, X)));
    local Z::AbstractVector = collect(k for k in bn.variables if (!haskey(e, k)));
    local state::Dict = copy(e);
    for Z_i in Z
        state[Z_i] = rand(RandomDeviceInstance, variable_values(bn, Z_i));
    end
    for j in 1:N
        for Z_i in Z
            state[Z_i] = markov_blanket_sample(Z_i, state, bn);
            counts[state[X]] = counts[state[X]] + 1;
        end
    end
    return ProbabilityDistribution(variable_name=X, frequencies=counts);
end

