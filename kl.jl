
# Learning with knowledge

export guess_example_value, generate_powerset, current_best_learning,
        version_space_learning,
        is_consistent_determination, minimal_consistent_determination,
        FOILKnowledgeBase, extend_example, choose_literal, new_literals,
        new_clause, foil;

function disjunction_value(e::Dict, d::Dict)
    for (k, v) in d
        if (!(typeof(v) <: AbstractString))
            error("disjunction_value(): Found an unexpected type, ", typeof(v), "!");
        end
        # Check for negation
        if (v[1] == '!')
            if (e[k] == v[2:end])
                return false;
            end
        elseif (e[k] != v)
            return false;
        end
    end
    return true;
end

"""
    guess_example_value(e::Dict, h::AbstractVector)

Return a guess for the logical value of the given example 'e' based on the given hypothesis 'h'.
"""
function guess_example_value(e::Dict, h::AbstractVector)
    for d in h
        if (disjunction_value(e, d))
            return true;
        end
    end
    return false;
end

function example_is_consistent(e::Dict, h::AbstractVector)
    return (e["GOAL"] == guess_example_value(e, h));
end

function example_is_false_positive(e::Dict, h::AbstractVector)
    if (e["GOAL"] == false)
        if (guess_example_value(e, h))
            return true;
        end
    end
    return false;
end

function example_is_false_negative(e::Dict, h::AbstractVector)
    if (e["GOAL"] == true)
        if (!(guess_example_value(e, h)))
            return true;
        end
    end
    return false;
end

function check_all_consistency(examples::AbstractVector, h::AbstractVector)
    for example in examples
        if (!(example_is_consistent(example, h)))
            return false;
        end
    end
    return true;
end

function specializations(prior_examples::AbstractVector, h::AbstractVector)
    local hypotheses::AbstractVector = [];
    for (i, disjunction) in enumerate(h)
        for example in prior_examples
            for (k, v) in example
                if ((haskey(disjunction, k)) || k == "GOAL")
                    continue;
                end

                local h_prime::Dict = copy(h[i]);
                h_prime[k] = "!" * v;
                local h_prime_prime::AbstractVector = copy(h);
                h_prime_prime[i] = h_prime;

                if (check_all_consistency(prior_examples, h_prime_prime))
                    push!(hypotheses, h_prime_prime);
                end
            end
        end
    end
    shuffle!(RandomDeviceInstance, hypotheses);
    return hypotheses;
end

function check_negative_consistency(examples::AbstractVector, h::Dict)
    for example in examples
        if (example["GOAL"])
            continue;
        end
        if (!example_is_consistent(example, [h]))
            return false;
        end
    end
    return true;
end

function generate_powerset(array::AbstractVector)
    local result::AbstractVector = Array{Any, 1}([()]);
    for element in array
        for i in eachindex(result)
            push!(result, (result[i]..., element));
        end
    end
    return Set{Tuple}(result);
end

function add_or_examples(prior_examples::AbstractVector, h::AbstractVector)
    local result::AbstractVector = [];
    local example::Dict = prior_examples[end];
    local attributes::Dict = Dict((k, v) for (k, v) in example if (k != "GOAL"));
    local attribute_powerset = setdiff!(generate_powerset(collect(keys(attributes))), Set([()]));

    for subset in attribute_powerset
        local h_prime::Dict = Dict();
        for key in subset
            h_prime[key] = attributes[key];
        end
        if (check_negative_consistency(prior_examples, h_prime))
            local h_prime_prime::AbstractVector = copy(h);
            push!(h_prime_prime, h_prime);
            push!(result, h_prime_prime);
        end
    end

    return result;
end

function generalizations(prior_examples::AbstractVector, h::AbstractVector)
    local hypotheses::AbstractVector = [];
    # Remove the empty set from the powerset.
    local disjunctions_powerset::Set = setdiff!(generate_powerset(collect(1:length(h))), Set([()]));
    for disjunctions in disjunctions_powerset
        h_prime = copy(h);
        deleteat!(h_prime, disjunctions);

        if (check_all_consistency(prior_examples, h_prime))
            append!(hypotheses, h_prime);
        end
    end

    for (i, disjunction) in enumerate(h)
        local attribute_powerset::Set = setdiff!(generate_powerset(collect(keys(disjunction))), Set([()]));
        for attributes in attribute_powerset
            h_prime = copy(h[i]);

            if (check_all_consistency(prior_examples, [h_prime]))
                local h_prime_prime::AbstractVector = copy(h);
                h_prime_prime[i] = copy(h_prime);
                push!(hypotheses, h_prime_prime);
            end
        end
    end
    if ((length(hypotheses) == 0) || (hypotheses == [Dict()]))
        hypotheses = add_or_examples(prior_examples, h);
    else
        append!(hypotheses, add_or_examples(prior_examples, h));
    end

    shuffle!(hypotheses);
    return hypotheses;
end

"""
    current_best_learning(examples::AbstractVector, h::AbstractVector, prior_examples::AbstractVector)
    current_best_learning(examples::AbstractVector, h::AbstractVector)

Apply the current-best-hypothesis learning algorithm (Fig. 19.2) on the given examples 'examples'
and hypothesis 'h' (an array of dictionaries where each Dict represents a disjunction). Return
a consistent hypothesis if possible, otherwise 'nothing' on failure.
"""
function current_best_learning(examples::AbstractVector, h::AbstractVector, prior_examples::AbstractVector)
    if (length(examples) == 0)
        return h;
    end

    local example::Dict = examples[1];

    push!(prior_examples, example);

    if (example_is_consistent(example, h))
        return current_best_learning(examples[2:end], h, prior_examples);
    elseif (example_is_false_positive(example, h))
        for h_prime in specializations(prior_examples, h)
            h_prime_prime = current_best_learning(examples[2:end], h_prime, prior_examples);
            if (!(typeof(h_prime_prime) <: Nothing))
                return h_prime_prime;
            end
        end
    elseif (example_is_false_negative(example, h))
        for h_prime in generalizations(prior_examples, h)
            h_prime_prime = current_best_learning(examples[2:end], h_prime, prior_examples);
            if (!(typeof(h_prime_prime) <: Nothing))
                return h_prime_prime;
            end
        end
    end
    return nothing;
end 

function current_best_learning(examples::AbstractVector, h::AbstractVector)
    return current_best_learning(examples, h, []);
end

function version_space_update(V::AbstractVector, e::Dict)
    return collect(h for h in V if (example_is_consistent(e, h)));
end

function values_table(examples::AbstractVector)
    local values::Dict = Dict();
    for example in examples
        for (k, v) in example
            if (k == "GOAL")
                continue
            end
            local modifier::String = "!";
            if (example["GOAL"])
                modifier = "";
            end

            local modified_value::String = modifier * v;
            if (!(modified_value in get!(values, k, [])))
                push!(get!(values, k, []), modified_value);
            end
        end
    end
    return values;
end

function build_attribute_combinations(subset::Tuple, values::Dict)
    local h::AbstractVector = [];
    if (length(subset) == 1)
        k = values[subset[1]]
        h = collect([Dict([Pair(subset[1], v)])] for v in values[subset[1]]);
        return h;
    end
    for (i, attribute) in enumerate(subset)
        local rest::AbstractVector = build_attribute_combinations(subset[2:end], values);
        for value in values[attribute]
            local combination::Dict = Dict([Pair(attribute, value)]);
            for rest_item in rest
                local combination_prime::Dict = copy(combination);
                for dictionary in rest_item
                    merge!(combination_prime, dictionary);
                end
                push!(h, [combination_prime]);
            end
        end
    end
    return h;
end

function build_h_combinations(hypotheses::AbstractVector)
    local h::AbstractVector = [];
    local h_powerset::Set = setdiff!(generate_powerset(collect(1:length(hypotheses))), Set([()]));

    for subset in h_powerset
        local combination::AbstractVector = [];
        for index in subset
            append!(combination, hypotheses[index]);
        end
        push!(h, combination);
    end

    return h;
end

function all_hypotheses(examples::AbstractVector)
    local values::Dict = values_table(examples);
    local h_powerset::Set = setdiff!(generate_powerset(collect(keys(values))), Set([()]));
    local hypotheses::AbstractVector = [];
    for subset in h_powerset
        append!(hypotheses, build_attribute_combinations(subset, values));
    end
    append!(hypotheses, build_h_combinations(hypotheses));
    return hypotheses;
end

"""
    version_space_learning(examples::AbstractVector)

Return a version space for the given 'examples' by using the version space learning
algorithm (Fig. 19.3).
"""
function version_space_learning(examples::AbstractVector)
    local V::AbstractVector = all_hypotheses(examples);
    for example in examples
        if (length(V) != 0)
            V = version_space_update(V, example);
        end
    end
    return V;
end

function is_consistent_determination(A::AbstractVector, E::AbstractVector)
    local H::Dict = Dict();

    for example in E
        local attribute_values::Tuple = Tuple((collect(example[attribute] for attribute in A)...,));
        if (haskey(H, attribute_values))
            if (H[attribute_values] != example["GOAL"])
                return false;
            end
        end
        H[attribute_values] = example["GOAL"];
    end

    return true;
end

"""
    minimal_consistent_determination(E::AbstractVector, A::Set)

Return a set of attributes by using the algorithm for finding a minimal consistent
determination (Fig. 19.8).
"""
function minimal_consistent_determination(E::AbstractVector, A::Set)
    local n::Int64 = length(A);
    for i in 0:n
        for A_i in combinations(A, i);
            if (is_consistent_determination(A_i, E))
                return Set(A_i);
            end
        end
    end
    return nothing;
end

#=

    FOILKnowledgeBase is a knowledge base that consists of first order logic definite clauses,

    constant symbols, and predicate symbols used by foil().

=#
mutable struct FOILKnowledgeBase <: AbstractKnowledgeBase
    fol_kb::FirstOrderLogicKnowledgeBase
    constant_symbols::Set
    predicate_symbols::Set

    function FOILKnowledgeBase()
        return new(FirstOrderLogicKnowledgeBase(), Set(), Set());
    end

    function FOILKnowledgeBase(initial_clauses::Array{Expression, 1})
        local fkb::FOILKnowledgeBase = new(FirstOrderLogicKnowledgeBase(), Set(), Set());
        for clause in initial_clauses
            tell(fkb, clause);
        end
        return fkb;
    end
end

function tell(fkb::FOILKnowledgeBase, e::Expression)
    if (!is_logic_definite_clause(e))
        error("tell(): ", repr(e), " , is not a definite clause!");
    end

    tell(fkb.fol_kb, e);
    fkb.constant_symbols = union(fkb.constant_symbols, constant_symbols(e));
    fkb.predicate_symbols = union(fkb.predicate_symbols, predicate_symbols(e));

    return nothing;
end

function ask(fkb::FOILKnowledgeBase, e::Expression)
    return fol_bc_ask(fkb.fol_kb, e);
end

function retract(fkb::FOILKnowledgeBase, e::Expression)
    retract(fkb.fol_kb, e);
    nothing;
end

"""
    extend_example(fkb::FOILKnowledgeBase, example::Dict, literal::Expression)

Return an array of extended examples by extending the given example 'example' to satisfy
the given literal 'literal'.
"""
function extend_example(fkb::FOILKnowledgeBase, example::Dict, literal::Expression)
    local solution::AbstractVector = [];
    local substitutions::Tuple = ask(fkb, substitute(example, literal));
    for substitution in substitutions
        push!(solution, merge!(substitution, example));
    end
    return solution;
end

"""
    update_positive_examples(fkb::FOILKnowledgeBase, examples_positive::AbstractVector, extended_positive_examples::AbstractVector, target::Expression)

Return an array of uncovered positive examples given the positive examples 'positive_examples' and
the extended positive examples 'extended_positive_examples'.
"""
function update_positive_examples(fkb::FOILKnowledgeBase, examples_positive::AbstractVector, extended_positive_examples::AbstractVector, target::Expression)
    local uncovered_positive_examples::Array{Dict, 1} = Array{Dict, 1}();
    for example in examples_positive
        if (any((function(dict::Dict)
                    return all((dict[x] == example[x]) for x in keys(example));
                end),
                extended_positive_examples))
            tell(fkb, substitute(example, target));
        else
            push!(uncovered_positive_examples, example);
        end
    end
    return uncovered_positive_examples;
end

"""
    new_literals(fkb::FOILKnowledgeBase, clause::Tuple{Expression, AbstractVector})

Return a Tuple of literals given the known predicate symbols in the FOIL knowledge base 'fkb'
and the horn clause 'clause'.

Each literal in the returned literals share at least 1 variable with the given horn clause.
"""
function new_literals(fkb::FOILKnowledgeBase, clause::Tuple{Expression, AbstractVector})
    local share_known_variables::Set = variables(clause[1]);
    for literal in clause[2]
        union!(share_known_variables, variables(literal));
    end
    local result::Tuple = ();
    for (predicate, arity) in fkb.predicate_symbols
        local new_variables::Set = Set(collect(standardize_variables(expr("x"), standardize_variables_counter)
                                                for i in 1:(arity - 1)));
        for arguments in iterable_cartesian_product(fill(union(share_known_variables, new_variables), arity))
            if (any((variable in share_known_variables) for variable in arguments))
                result = Tuple((result..., Expression(predicate, arguments...,)));
            end
        end
    end
    return result;
end

"""
    choose_literal(fkb::FOILKnowledgeBase, literals::Tuple, examples::Tuple{AbstractVector, AbstractVector})

Return the best literal from the given literals 'literals' by comparing the information gained.
"""
function choose_literal(fkb::FOILKnowledgeBase, literals::Tuple, examples::Tuple{AbstractVector, AbstractVector})
    local information_gain::Function = (function(literal::Expression)
                                            local examples_positive::Int64 = length(examples[1]);
                                            local examples_negative::Int64 = length(examples[2]);
                                            local extended_examples::AbstractVector = collect(vcat(collect(extend_example(fkb, example, literal)
                                                                                                            for example in examples[i])...,)
                                                                                            for i in 1:2);
                                            local extended_examples_positive::Int64 = length(extended_examples[1]);
                                            local extended_examples_negative::Int64 = length(extended_examples[2]);
                                            if ((examples_positive + examples_negative == 0) ||
                                                (extended_examples_positive + extended_examples_negative == 0))
                                                return (literal, -1);
                                            end
                                            local T::Int64 = 0;
                                            for example in examples[1]
                                                if (any((function(l_prime::Dict)
                                                            return all((l_prime[x] == example[x]) for x in keys(example));
                                                        end),
                                                        extended_examples[1]))
                                                    T = T + 1;
                                                end
                                            end
                                            return (literal, (T * log((extended_examples_positive * (examples_positive + examples_negative) + 0.0001)/((extended_examples_positive + extended_examples_negative) * examples_positive))));
                                        end);

    local gains::Tuple = map(information_gain, literals);
    return reduce((function(t1::Tuple, t2::Tuple)
                        if (getindex(t1, 2) < getindex(t2, 2))
                            return t2;
                        else
                            return t1;
                        end
                    end), gains)[1];
end

"""
    new_clause(fkb::FOILKnowledgeBase, examples::Tuple{AbstractVector, AbstractVector}, target::Expression)

Return a horn clause and the extended positive examples as Tuple.

The horn clause is represented as (consequent, array of antecendents).
"""
function new_clause(fkb::FOILKnowledgeBase, examples::Tuple{AbstractVector, AbstractVector}, target::Expression)
    local clause::Tuple = (target, Array{Expression, 1}());
    extended_examples = examples;
    while (length(extended_examples[2]) != 0)
        local literal::Expression = choose_literal(fkb, new_literals(fkb, clause), extended_examples);
        push!(clause[2], literal);
        extended_examples = (collect(vcat(collect(extend_example(fkb, example, literal)
                                                for example in extended_examples[i])...,)
                                    for i in 1:2)...,);
    end
    return (clause, extended_examples[1]);
end

"""
    foil(fkb::FOILKnowledgeBase, examples::Tuple{AbstractVector, AbstractVector}, target::Expression)

Return an array of horn clauses by using the FOIL algorithm (Fig. 19.12) on the given FOIL knowledge
base 'fkb', set of examples 'examples', and the target literal 'target'.
"""
function foil(fkb::FOILKnowledgeBase, examples::Tuple{AbstractVector, AbstractVector}, target::Expression)
    local clauses::AbstractVector = [];
    local positive_examples::AbstractVector;
    local negative_examples::AbstractVector;
    positive_examples, negative_examples = examples;

    while (length(positive_examples) != 0)
        local clause::Tuple;
        local positive_extended_examples::AbstractVector;
        clause, positive_extended_examples = new_clause(fkb, (positive_examples, negative_examples), target);
        # Remove postive examples covered by 'clause' from 'examples'
        positive_examples = update_positive_examples(fkb, positive_examples, positive_extended_examples, target);
        push!(clauses, clause);
    end
    return clauses;
end

