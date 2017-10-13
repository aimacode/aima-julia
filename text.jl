
export extract_words, canonicalize_text, UnigramWordModel, NgramWordModel, samples;

#=

    UnigramWordModel is a probability distribution for counting observations of words.

=#
type UnigramWordModel
    dict::Dict
    number_of_observations::Int64
    default::Int64
    sample_function::Nullable{Function}

    function UnigramWordModel(observations::AbstractVector; default::Int64=0)
        local uwm::UnigramWordModel = new(Dict(), 0, default, Nullable{Function}());
        for observation in observations
            add(uwm, observation);
        end
        return uwm;
    end

    function UnigramWordModel(; default::Int64=0)
        local uwm::UnigramWordModel = new(Dict(), 0, default, Nullable{Function}());
        return uwm;
    end
end

function add(uwm::UnigramWordModel, observation)
    smooth_for_observation(uwm, observation);
    uwm.dict[observation] = uwm.dict[observation] + 1;
    uwm.number_of_observations = uwm.number_of_observations + 1;
    uwm.sample_function = Nullable{Function}();
    nothing;
end

function smooth_for_observation(uwm::UnigramWordModel, observation)
    if (!(observation in keys(uwm.dict)))
        uwm.dict[observation] = uwm.default;
        uwm.number_of_observations = uwm.number_of_observations + uwm.default;
        uwm.sample_function = Nullable{Function}();
    end
    nothing;
end

"""
    getindex(uwm::UnigramWordModel, key)

Return the probability of the given 'key'.
"""
function getindex(uwm::UnigramWordModel, key)
    smooth_for_observation(uwm, key);
    return (Float64(uwm.dict[key]) / Float64(uwm.number_of_observations));
end

"""
    top(uwm::UnigramWordModel, n::Int64)

Return an array of (observation_count, observation) tuples such that the array
does not exceed length 'n'.
"""
function top(uwm::UnigramWordModel, n::Int64)
    local observations::AbstractVector = sort(collect(reverse((i...)) for i in uwm.dict),
                                                lt=(function(p1::Tuple{Number, Any}, p2::Tuple{Number, Any})
                                                        return (p1[1] > p2[1]);
                                                    end));
    if (length(observations) <= n)
        return observations;
    else
        return observations[1:n];
    end
end

"""
    sample(uwm::UnigramWordModel)

Return a random sample from the probability distribution 'uwm'.
"""
function sample(uwm::UnigramWordModel)
    if (isnull(uwm.sample_function))
        uwm.sample_function = weighted_sampler(collect(keys(uwm.dict)), collect(values(uwm.dict)));
    end
    return get(uwm.sample_function)();
end

"""
    samples(uwm::UnigramWordModel, n::Int64)

Return a String of 'n' words by using 'n' random samples made from the probability distribution 'uwm'.
"""
function samples(uwm::UnigramWordModel, n::Int64)
    return join(collect(sample(uwm) for i in 1:n), " ");
end

#=

    NgramWordModel is a probability distribution for counting observations of words.

=#
type NgramWordModel
    dict::Dict
    number_of_observations::Int64
    default::Int64
    sample_function::Nullable{Function}
    n::Int64
    conditional_probabilities::Dict

    function NgramWordModel(n::Int64, observations::AbstractVector; default::Int64=0)
        local nwm::NgramWordModel = new(Dict(), 0, default, Nullable{Function}(), n, Dict());
        add_sequence(nwm, observations);
        return nwm;
    end
end

function add(nwm::NgramWordModel, observation)
    smooth_for_observation(nwm, observation);
    nwm.dict[observation] = nwm.dict[observation] + 1;
    nwm.number_of_observations = nwm.number_of_observations + 1;
    nwm.sample_function = Nullable{Function}();
    nothing;
end

function smooth_for_observation(nwm::NgramWordModel, observation)
    if (!(observation in keys(nwm.dict)))
        nwm.dict[observation] = nwm.default;
        nwm.number_of_observations = nwm.number_of_observations + nwm.default;
        nwm.sample_function = Nullable{Function}();
    end
    nothing;
end

"""
    getindex(nwm::NgramWordModel, key)

Return the probability of the given 'key'.
"""
function getindex(nwm::NgramWordModel, key)
    smooth_for_observation(nwm, key);
    return (Float64(nwm.dict[key]) / Float64(nwm.number_of_observations));
end

"""
    top(nwm::NgramWordModel, n::Int64)

Return an array of (observation_count, observation) tuples such that the array
does not exceed length 'n'.
"""
function top(nwm::NgramWordModel, n::Int64)
    local observations::AbstractVector = sort(collect(reverse((i...)) for i in nwm.dict),
                                                lt=(function(p1::Tuple{Number, Any}, p2::Tuple{Number, Any})
                                                        return (p1[1] > p2[1]);
                                                    end));
    if (length(observations) <= n)
        return observations;
    else
        return observations[1:n];
    end
end

"""
    sample(nwm::NgramWordModel)

Return a random sample from the probability distribution 'nwm'.
"""
function sample(nwm::NgramWordModel)
    if (isnull(nwm.sample_function))
        nwm.sample_function = weighted_sampler(collect(keys(nwm.dict)), collect(values(nwm.dict)));
    end
    return get(nwm.sample_function)();
end

function add_conditional_probability(nwm::NgramWordModel, ngram::Tuple)
    if (!(haskey(nwm.conditional_probabilities, ngram[1:(end - 1)])))
        nwm.conditional_probabilities[ngram[1:(end - 1)]] = CountingProbabilityDistribution();
    end
    add(nwm.conditional_probabilities[ngram[1:(end - 1)]], ngram[end]);
end

function add_sequence(nwm::NgramWordModel, words::AbstractVector)
    for i in 1:(length(words) - nwm.n + 1)
        local t::Tuple = Tuple((words[i:(i + nwm.n - 1)]...));
        add(nwm, t);
        add_conditional_probability(nwm, t);
    end
    return nothing;
end

function samples(nwm::NgramWordModel, n::Int64)
    local output::AbstractVector = collect(sample(nwm));
    for i in nwm.n+1:n
        local start_index::Int64 = length(output) - nwm.n + 2;
        local last::Tuple = Tuple((output[start_index:end]...));
        local next_word::String = sample(nwm.conditional_probabilities[last]);
        push!(output, next_word);
    end
    return join(output, " ");
end

"""
    extract_words(str::String)

Return an Array of lowercase alphanumeric Strings.
"""
function extract_words(str::String)
    return map(lowercase, matchall(@r_str("[a-zA-Z0-9]+"), str));
end

"""
    canonicalize_text(str::String)

Return a String from the given string 'str' with only blanks and lowercase letters.
"""
function canonicalize_text(str::String)
    return join(extract_words(str), " ");
end

