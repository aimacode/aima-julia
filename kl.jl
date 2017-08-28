
# Learning with knowledge

export generate_powerset, current_best_learning;

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

function guess_example_value(e::Dict, h::AbstractVector)
    for d in h
        if (disjunction_value(e, d))
            return true;
        end
    end
    return false;
end

function is_consistent(e::Dict, h::AbstractVector)
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
        if (!(is_consistent(example, h)))
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
                if ((k in disjunction) || k == "GOAL")
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
        if (!is_consistent(example, [h]))
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

function and_or_examples(prior_examples::AbstractVector, h::AbstractVector)
    local result::AbstractVector;
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
        deleteat!(h2, disjunctions);

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
            if (!(typeof(h_prime_prime) <: Void))
                return h_prime_prime;
            end
        end
    elseif (example_is_false_negative(example, h))
        for h_prime in generalizations(prior_examples, h)
            h_prime_prime = current_best_learning(examples[2:end], h_prime, prior_examples);
            if (!(typeof(h_prime_prime) <: Void))
                return h_prime_prime;
            end
        end
    end
    return nothing;
end 

function current_best_learning(examples::AbstractVector, h::AbstractVector)
    return current_best_learning(examples, h, []);
end


