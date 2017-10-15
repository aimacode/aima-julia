
export extract_words, canonicalize_text, UnigramWordModel, NgramWordModel, samples,
        UnigramCharModel, NgramCharModel,
        shift_encode, rot13, bigrams;

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

"""
    add(uwm::UnigramWordModel, observation::String)

Add observation 'observation' to the probability distribution 'uwm'.
"""
function add(uwm::UnigramWordModel, observation::String)
    smooth_for_observation(uwm, observation);
    uwm.dict[observation] = uwm.dict[observation] + 1;
    uwm.number_of_observations = uwm.number_of_observations + 1;
    uwm.sample_function = Nullable{Function}();
    nothing;
end

"""
    smooth_for_observation(uwm::UnigramWordModel, observation::String)

Initialize observation 'observation' in the distribution 'uwm' if the observation doesn't
exist in the distribution yet.
"""
function smooth_for_observation(uwm::UnigramWordModel, observation::String)
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

    NgramWordModel is a probability distribution for counting observations of n-grams

    consisting of consecutive words.

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

"""
    add(nwm::NgramWordModel, observation::Tuple)

Add observation 'observation' to the probability distribution 'nwm'.
"""
function add(nwm::NgramWordModel, observation::Tuple)
    smooth_for_observation(nwm, observation);
    nwm.dict[observation] = nwm.dict[observation] + 1;
    nwm.number_of_observations = nwm.number_of_observations + 1;
    nwm.sample_function = Nullable{Function}();
    nothing;
end

"""
    smooth_for_observation(nwm::NgramWordModel, observation::Tuple)

Initialize observation 'observation' in the distribution 'nwm' if the observation doesn't
exist in the distribution yet.
"""
function smooth_for_observation(nwm::NgramWordModel, observation::Tuple)
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

"""
    add_conditional_probability(nwm::NgramWordModel, ngram::Tuple)

Add the conditional probability distribution P(w_n | (w_1, w_2, ..., w_{n-1})) for the observation 'ngram'
to the dictionary of conditional probabilities for distribution 'nwm'.
"""
function add_conditional_probability(nwm::NgramWordModel, ngram::Tuple)
    if (!(haskey(nwm.conditional_probabilities, ngram[1:(end - 1)])))
        nwm.conditional_probabilities[ngram[1:(end - 1)]] = CountingProbabilityDistribution();
    end
    add(nwm.conditional_probabilities[ngram[1:(end - 1)]], ngram[end]);
end

"""
    add_sequence(nwm::NgramWordModel, words::AbstractVector)

Add Tuples of 'n' consecutive characters as observations for probability distribution 'nwm'.
"""
function add_sequence(nwm::NgramWordModel, words::AbstractVector)
    for i in 1:(length(words) - nwm.n + 1)
        local t::Tuple = Tuple((words[i:(i + nwm.n - 1)]...));
        add(nwm, t);
        add_conditional_probability(nwm, t);
    end
    return nothing;
end

"""
    samples(nwm::NgramWordModel, n::Int64)

Return a String of 'n' words by using 'n' random samples made from the probability distribution 'uwm'.

The first nwm.n words are from a randomly selected 'nwm.n'-gram. The following words are randomly
selected from the probability distribution P(c | w_{l - 1}, w_{l - 2}, ..., w_{l - n + 1}) such that
w_{l - 1}, w_{l - 2}, ..., w_{l - n + 1} are the last 'n' - 1 words of the generated sentence.
"""
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

#=

    UnigramCharModel is a probability distribution for counting observations of characters (letters).

=#
type UnigramCharModel
    dict::Dict
    number_of_observations::Int64
    default::Int64
    sample_function::Nullable{Function}

    function UnigramCharModel(observations::AbstractVector; default::Int64=0)
        local ucm::UnigramCharModel = new(Dict(), 0, default, Nullable{Function}());
        add_sequence(ucm, observations);
        return ucm;
    end
end

"""
    add(ucm::UnigramCharModel, observation::Char)

Add observation 'observation' to the probability distribution 'ucm'.
"""
function add(ucm::UnigramCharModel, observation::Char)
    smooth_for_observation(ucm, observation);
    ucm.dict[observation] = ucm.dict[observation] + 1;
    ucm.number_of_observations = ucm.number_of_observations + 1;
    ucm.sample_function = Nullable{Function}();
    nothing;
end

"""
    smooth_for_observation(ucm::UnigramCharModel, observation::Char)

Initialize observation 'observation' in the distribution 'ucm' if the observation doesn't
exist in the distribution yet.
"""
function smooth_for_observation(ucm::UnigramCharModel, observation::Char)
    if (!(observation in keys(ucm.dict)))
        ucm.dict[observation] = ucm.default;
        ucm.number_of_observations = ucm.number_of_observations + ucm.default;
        ucm.sample_function = Nullable{Function}();
    end
    nothing;
end

"""
    getindex(ucm::UnigramCharModel, key)

Return the probability of the given 'key'.
"""
function getindex(ucm::UnigramCharModel, key)
    smooth_for_observation(ucm, key);
    return (Float64(ucm.dict[key]) / Float64(ucm.number_of_observations));
end

"""
    top(ucm::UnigramCharModel, n::Int64)

Return an array of (observation_count, observation) tuples such that the array
does not exceed length 'n'.
"""
function top(ucm::UnigramCharModel, n::Int64)
    local observations::AbstractVector = sort(collect(reverse((i...)) for i in ucm.dict),
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
    sample(ucm::UnigramCharModel)

Return a random sample from the probability distribution 'ucm'.
"""
function sample(ucm::UnigramCharModel)
    if (isnull(ucm.sample_function))
        ucm.sample_function = weighted_sampler(collect(keys(ucm.dict)), collect(values(ucm.dict)));
    end
    return get(ucm.sample_function)();
end

"""
    add_sequence(ucm::UnigramCharModel, words::AbstractVector)

Add the characters (letters) of the words 'words' as observations for probability distribution 'ucm'.
"""
function add_sequence(ucm::UnigramCharModel, words::AbstractVector)
    for word in words
        for i in 1:length(word)
            add(ucm, word[i]);
        end
    end
    return nothing;
end

"""
    samples(uwm::UnigramWordModel, n::Int64)

Return a String of 'n' characters by using 'n' random samples made from the probability distribution 'uwm'.
"""
function samples(uwm::UnigramCharModel, n::Int64)
    return String(collect(sample(uwm) for i in 1:n));
end

#=

    NgramCharModel is a probability distribution for counting observations of n-grams

    consisting of consecutive characters.

=#
type NgramCharModel
    dict::Dict
    number_of_observations::Int64
    default::Int64
    sample_function::Nullable{Function}
    n::Int64
    conditional_probabilities::Dict

    function NgramCharModel(n::Int64, observations::AbstractVector; default::Int64=0)
        local ncm::NgramCharModel = new(Dict(), 0, default, Nullable{Function}(), n, Dict());
        add_sequence(ncm, observations);
        return ncm;
    end
end

"""
    add(ncm::NgramCharModel, observation::Tuple)

Add observation 'observation' to the probability distribution 'ncm'.
"""
function add(ncm::NgramCharModel, observation::Tuple)
    smooth_for_observation(ncm, observation);
    ncm.dict[observation] = ncm.dict[observation] + 1;
    ncm.number_of_observations = ncm.number_of_observations + 1;
    ncm.sample_function = Nullable{Function}();
    nothing;
end

"""
    smooth_for_observation(ncm::NgramCharModel, observation::Tuple)

Initialize observation 'observation' in the distribution 'ncm' if the observation doesn't
exist in the distribution yet.
"""
function smooth_for_observation(ncm::NgramCharModel, observation::Tuple)
    if (!(observation in keys(ncm.dict)))
        ncm.dict[observation] = ncm.default;
        ncm.number_of_observations = ncm.number_of_observations + ncm.default;
        ncm.sample_function = Nullable{Function}();
    end
    nothing;
end

"""
    getindex(ncm::NgramCharModel, key)

Return the probability of the given 'key'.
"""
function getindex(ncm::NgramCharModel, key)
    smooth_for_observation(ncm, key);
    return (Float64(ncm.dict[key]) / Float64(ncm.number_of_observations));
end

"""
    top(ncm::NgramCharModel, n::Int64)

Return an array of (observation_count, observation) tuples such that the array
does not exceed length 'n'.
"""
function top(ncm::NgramCharModel, n::Int64)
    local observations::AbstractVector = sort(collect(reverse((i...)) for i in ncm.dict),
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
    sample(ncm::NgramCharModel)

Return a random sample from the probability distribution 'ncm'.
"""
function sample(ncm::NgramCharModel)
    if (isnull(ncm.sample_function))
        ncm.sample_function = weighted_sampler(collect(keys(ncm.dict)), collect(values(ncm.dict)));
    end
    return get(ncm.sample_function)();
end

"""
    add_conditional_probability(ncm::NgramCharModel, ngram::Tuple)

Add the conditional probability distribution P(w_n | (w_1, w_2, ..., w_{n-1})) for the observation 'ngram'
to the dictionary of conditional probabilities for distribution 'ncm'.
"""
function add_conditional_probability(ncm::NgramCharModel, ngram::Tuple)
    if (!(haskey(ncm.conditional_probabilities, ngram[1:(end - 1)])))
        ncm.conditional_probabilities[ngram[1:(end - 1)]] = CountingProbabilityDistribution();
    end
    add(ncm.conditional_probabilities[ngram[1:(end - 1)]], ngram[end]);
end

"""
    add_sequence(ncm::NgramCharModel, words::AbstractVector)

Add Tuples of 'n' consecutive characters as observations for probability distribution 'ncm'.
"""
function add_sequence(ncm::NgramCharModel, words::AbstractVector)
    for word in map(*, repeated(" ", length(words)), words)
        for i in 1:(length(word) - ncm.n + 1)
            local t::Tuple = Tuple((word[i:(i + ncm.n - 1)]...));
            add(ncm, t);
            add_conditional_probability(ncm, t);
        end
    end
    return nothing;
end

"""
    samples(ncm::NgramCharModel, n::Int64)

Return a String of 'n' characters by using 'n' random samples made from the probability distribution 'uwm'.

The first nwm.n words are from a randomly selected 'nwm.n'-gram. The following characters are randomly
selected from the probability distribution P(c | w_{l - 1}, w_{l - 2}, ..., w_{l - n + 1}) such that
w_{l - 1}, w_{l - 2}, ..., w_{l - n + 1} are the last 'n' - 1 characters of the generated string.
"""
function samples(ncm::NgramCharModel, n::Int64)
    local output::AbstractVector = collect(sample(ncm));
    for i in ncm.n+1:n
        local start_index::Int64 = length(output) - ncm.n + 2;
        local last::Tuple = Tuple((output[start_index:end]...));
        local next_word::String = sample(ncm.conditional_probabilities[last]);
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

lowercase_alphabet = "abcdefghijklmnopqrstuvwxyz";

"""
    generate_translation_table(from::String, to::String)

Return a function that uses a translation table generated from the given arrays 'from' and 'to'.
"""
function generate_translation_table(from::String, to::String)
    local translation_dict::Dict = Dict();
    for (i, character) in enumerate(from)
        translation_dict[character] = to[i];
    end
    return (function(character::Char)
                return get(translation_dict, character, character);
            end);
end

"""
    encode_text(plaintext::String, code::String)

Return the text encoded by a substitution cipher given a permutation of the alphabet 'code'.
"""
function encode_text(plaintext::String, code::String)
    local translations::Function = generate_translation_table(lowercase_alphabet * uppercase(lowercase_alphabet), code * uppercase(code));
    return String(map(translations, collect(plaintext)));
end

"""


Return the encoded text by using a shift cipher (Caesar cipher) that rotates the alphabet by 'n' letters.
"""
function shift_encode(plaintext::String, n::Int64)
    return encode_text(plaintext, lowercase_alphabet[(n + 1):end] * lowercase_alphabet[1:n])
end

"""
    rot13(plaintext::String)

Return the encoded text by rotating letters by 13 places in the alphabet.
"""
function rot13(plaintext::String)
    return shift_encode(plaintext, 13);
end

"""
    bigrams(text::String)

Return an array of 2 character Strings of consisting of adjacent letters in the given String 'text'.
"""
function bigrams(text::String)
    return collect(Tuple((text[i:(i + 1)]...)) for i in 1:(length(text) - 1));
end

"""
    bigrams(text::AbstractVector)

Return an array of 2 word Tuples of consisting of adjacent words in the given array 'text'.
"""
function bigrams(text::AbstractVector)
    return collect(Tuple((text[i:(i + 1)]...)) for i in 1:(length(text) - 1));
end

