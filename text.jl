
export extract_words, canonicalize_text, UnigramWordModel, NgramWordModel, samples,
        UnigramCharModel, NgramCharModel,
        shift_encode, rot13, bigrams, viterbi_text_segmentation,
        DocumentMetadata,
        AbstractInformationRetrievalSystem, InformationRetrievalSystem, UnixConsultant, execute_query;

#=

    UnigramWordModel is a probability distribution for counting observations of words.

=#
type UnigramWordModel <: AbstractCountingProbabilityDistribution
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
type NgramWordModel <: AbstractCountingProbabilityDistribution
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
type UnigramCharModel <: AbstractCountingProbabilityDistribution
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
    samples(uwm::UnigramCharModel, n::Int64)

Return a String of 'n' characters by using 'n' random samples made from the probability distribution 'uwm'.
"""
function samples(uwm::UnigramCharModel, n::Int64)
    return String(collect(sample(uwm) for i in 1:n));
end

#=

    NgramCharModel is a probability distribution for counting observations of n-grams

    consisting of consecutive characters.

=#
type NgramCharModel <: AbstractCountingProbabilityDistribution
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

"""
    viterbi_text_segmentation(text::String, P::UnigramWordModel)

Return the best segmentation of the given text 'text' as an array of Strings and its corresponding
probability by applying the Viterbi algorithm on the given text 'text' and probabiliy distribution 'P'.
"""
function viterbi_text_segmentation(text::String, P::UnigramWordModel)
    # words[i] - best word ending at index (i - 1)
    # best[i] - best probability for text[1:(i - 1)]
    local words::AbstractVector = vcat([""], map(String, collect([c] for c in text)));
    local n::Int64 = length(text);
    local best::AbstractVector = vcat([1.0], fill(0.0, n));

    # Update 'words' and 'best' if a better word is found.
    for i in 1:(n + 1)
        for j in 1:i
            local w::String = text[j:(i - 1)];
            local current_score::Float64 = P[w] * best[(i - length(w))];
            if (current_score >= best[i])
                best[i] = current_score;
                words[i] = w;
            end
        end
    end

    # Reconstruct the Viterbi path for the best segmentation of the given text.
    local sequence::AbstractVector = [];
    local idx::Int64 = length(words);
    while (idx > 1)     # Julia uses 1-indexing
        unshift!(sequence, words[idx]);
        idx = idx - length(words[idx]);
    end
    return sequence, best[end];
end

#=

    DocumentMetadata is the metadata for a document.

=#
type DocumentMetadata
    title::String
    url::String
    number_of_words::Int64

    function DocumentMetadata(title::String, url::String, number_of_words::Int64)
        return new(title, url, number_of_words);
    end
end

abstract AbstractInformationRetrievalSystem;

#=

    InformationRetrievalSystem is a information retrieval (IR) system implementation

    that consists of an index, a set of stop words, and the metadata for the documents.

=#
type InformationRetrievalSystem <: AbstractInformationRetrievalSystem
    index::Dict
    stop_words::Set
    documents::AbstractVector

    function InformationRetrievalSystem(stop_words::String)
        return new(Dict(), Set(extract_words(stop_words)), []);
    end

    function InformationRetrievalSystem()
        return new(Dict(), Set(["the", "a", "of"]), []);
    end
end

"""
    index_document{T <: AbstractInformationRetrievalSystem}(irs::T, text::String, url::String)

Index the document by its text 'text' and URL 'url'.
"""
function index_document{T <: AbstractInformationRetrievalSystem}(irs::T, text::String, url::String)
    local title::String = strip(text[1:(Base.search(text, "\n").stop)]);
    local document_words::AbstractVector = extract_words(text);
    push!(irs.documents, DocumentMetadata(title, url, length(document_words)));
    local document_id::Int64 = length(irs.documents);
    for word in document_words
        if (!(word in irs.stop_words))
            get!(irs.index, word, Dict())[document_id] = get!(get!(irs.index, word, Dict()), document_id, 0) + 1;
        end
    end
    return nothing;
end

"""
    index_collection{T <: AbstractInformationRetrievalSystem}(irs::T, filenames::AbstractVector)

Index the given collection of files 'filenames'.
"""
function index_collection{T <: AbstractInformationRetrievalSystem}(irs::T, filenames::AbstractVector)
    for filename in filenames
        index_document(irs, String(read(filename)), relpath(filename, AIMAJULIA_DIRECTORY));
    end
    return nothing;
end

"""
    score_document{T <: AbstractInformationRetrievalSystem}(irs::T, word::String, document_id::Int64)

Return a score for the given word 'word' and document referenced by ID 'document_id'.
"""
function score_document{T <: AbstractInformationRetrievalSystem}(irs::T, word::String, document_id::Int64)
    return (log(1 + get!(get!(irs.index, word, Dict()), document_id, 0)) / log(1 + irs.documents[document_id].number_of_words));
end

"""
    total_score_document{T <: AbstractInformationRetrievalSystem}(irs::T, words::AbstractVector, document_id::Int64)

Return the sum of scores for the given words 'words' within the document referenced by ID 'document_id'.
"""
function total_score_document{T <: AbstractInformationRetrievalSystem}(irs::T, words::AbstractVector, document_id::Int64)
    return sum(score_document(irs, word, document_id) for word in words);
end

"""
    execute_query{T <: AbstractInformationRetrievalSystem}(irs::T, query::String; n::Int64=10)

Return an array of 'n' (score, document ID) Tuples for the best matches.

If the query starts with 'learn: ', the following command within the query is executed.
Then the command output is then indexed with index_document().
"""
function execute_query{T <: AbstractInformationRetrievalSystem}(irs::T, query::String; n::Int64=10)
    if (startswith(query, "learn:"))
        local truncated_query::String = strip(query[7:end]);
        local document_text::String = strip(readstring(`$truncated_query`));
        index_document(irs, document_text, query);
        return [];
    end
    local query_words::AbstractVector = collect(word for word in extract_words(query)
                                                if (!(word in irs.stop_words)));
    local shortest_word::String = argmin(query_words,
                                        (function(s::String)
                                            return length(get!(irs.index, s, Dict()));
                                        end));
    local document_ids::AbstractVector = collect(keys(get!(irs.index, shortest_word, Dict())));
    local document_ids_scores::AbstractVector = sort(collect((total_score_document(irs, query_words, id), id) for id in document_ids),
                                                    lt=(function(p1::Tuple{Number, Any}, p2::Tuple{Number, Any})
                                                            return (p1[1] > p2[1]);
                                                        end));
    if (length(document_ids_scores) <= n)
        return document_ids_scores;
    else
        return document_ids_scores[1:n];
    end
end

#=

    UnixConsultant is a information retrieval (IR) system implementation for Unix man (manual)

    pages which consists of an index, a set of stop words, and the metadata for the documents.

=#
type UnixConsultant <: AbstractInformationRetrievalSystem
    index::Dict
    stop_words::Set
    documents::AbstractVector

    function UnixConsultant(stop_words::String)
        local uc::UnixConsultant = new(Dict(), Set(extract_words(stop_words)), []);
        index_collection(uc, collect(joinpath(joinpath(AIMAJULIA_DIRECTORY, "aima-data"), filename)
                                    for filename in readdir(joinpath(AIMAJULIA_DIRECTORY, "aima-data"))
                                    if (endswith(filename, ".txt"))));
        return uc;
    end

    function UnixConsultant()
        local uc::UnixConsultant = new(Dict(), Set(["how", "do", "i", "the", "a", "of"]), []);
        index_collection(uc, collect(joinpath(joinpath(joinpath(AIMAJULIA_DIRECTORY, "aima-data"), "MAN"), filename)
                                    for filename in readdir(joinpath(joinpath(AIMAJULIA_DIRECTORY, "aima-data"), "MAN"))
                                    if (endswith(filename, ".txt"))));
        return uc;
    end
end

