
export extract_words, canonicalize_text, UnigramWordModel, NgramWordModel, samples,
        UnigramCharModel, NgramCharModel,
        shift_encode, rot13, bigrams, viterbi_text_segmentation,
        DocumentMetadata,
        AbstractInformationRetrievalSystem, InformationRetrievalSystem, UnixConsultant, execute_query,
        ShiftCipherDecoder, score_text, decode_text,
        PermutationCipherDecoder, PermutationCipherDecoderProblem;

#=

    UnigramWordModel is a probability distribution for counting observations of words.

=#
mutable struct UnigramWordModel <: AbstractCountingProbabilityDistribution
    dict::Dict
    number_of_observations::Int64
    default::Int64
    sample_function::Union{Nothing, Function}

    function UnigramWordModel(observations::AbstractVector; default::Int64=0)
        local uwm::UnigramWordModel = new(Dict(), 0, default, nothing);
        for observation in observations
            add(uwm, observation);
        end
        return uwm;
    end

    function UnigramWordModel(; default::Int64=0)
        local uwm::UnigramWordModel = new(Dict(), 0, default, nothing);
        return uwm;
    end
end

"""
    sample(uwm::UnigramWordModel)

Return a random sample from the probability distribution 'uwm'.
"""
function sample(uwm::UnigramWordModel)
    if (uwm.sample_function === nothing)
        uwm.sample_function = weighted_sampler(collect(keys(uwm.dict)), collect(values(uwm.dict)));
    end
    return uwm.sample_function();
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
mutable struct NgramWordModel <: AbstractCountingProbabilityDistribution
    dict::Dict
    number_of_observations::Int64
    default::Int64
    sample_function::Union{Nothing, Function}
    n::Int64
    conditional_probabilities::Dict

    function NgramWordModel(n::Int64, observations::AbstractVector; default::Int64=0)
        local nwm::NgramWordModel = new(Dict(), 0, default, nothing, n, Dict());
        add_sequence(nwm, observations);
        return nwm;
    end
end

"""
    sample(nwm::NgramWordModel)

Return a random sample from the probability distribution 'nwm'.
"""
function sample(nwm::NgramWordModel)
    if (nwm.sample_function === nothing)
        nwm.sample_function = weighted_sampler(collect(keys(nwm.dict)), collect(values(nwm.dict)));
    end
    return nwm.sample_function();
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
        local t::Tuple = Tuple((words[i:(i + nwm.n - 1)]...,));
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
        local last::Tuple = Tuple((output[start_index:end]...,));
        local next_word::String = sample(nwm.conditional_probabilities[last]);
        push!(output, next_word);
    end
    return join(output, " ");
end

#=

    UnigramCharModel is a probability distribution for counting observations of characters (letters).

=#
mutable struct UnigramCharModel <: AbstractCountingProbabilityDistribution
    dict::Dict
    number_of_observations::Int64
    default::Int64
    sample_function::Union{Nothing, Function}

    function UnigramCharModel(observations::AbstractVector; default::Int64=0)
        local ucm::UnigramCharModel = new(Dict(), 0, default, nothing);
        add_sequence(ucm, observations);
        return ucm;
    end
end

"""
    sample(ucm::UnigramCharModel)

Return a random sample from the probability distribution 'ucm'.
"""
function sample(ucm::UnigramCharModel)
    if (ucm.sample_function === nothing)
        ucm.sample_function = weighted_sampler(collect(keys(ucm.dict)), collect(values(ucm.dict)));
    end
    return ucm.sample_function();
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
mutable struct NgramCharModel <: AbstractCountingProbabilityDistribution
    dict::Dict
    number_of_observations::Int64
    default::Int64
    sample_function::Union{Nothing, Function}
    n::Int64
    conditional_probabilities::Dict

    function NgramCharModel(n::Int64, observations::AbstractVector; default::Int64=0)
        local ncm::NgramCharModel = new(Dict(), 0, default, nothing, n, Dict());
        add_sequence(ncm, observations);
        return ncm;
    end
end

"""
    sample(ncm::NgramCharModel)

Return a random sample from the probability distribution 'ncm'.
"""
function sample(ncm::NgramCharModel)
    if (ncm.sample_function === nothing)
        ncm.sample_function = weighted_sampler(collect(keys(ncm.dict)), collect(values(ncm.dict)));
    end
    return ncm.sample_function();
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
    for word in map(*, Base.Iterators.repeated(" ", length(words)), words)
        for i in 1:(length(word) - ncm.n + 1)
            local t::Tuple = Tuple((word[i:(i + ncm.n - 1)]...,));
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
        local last::Tuple = Tuple((output[start_index:end]...,));
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
    return map(lowercase, collect(m.match for m in eachmatch(@r_str("[a-zA-Z0-9]+"), str)));
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
    return collect(Tuple((text[i:(i + 1)]...,)) for i in 1:(length(text) - 1));
end

"""
    bigrams(text::AbstractVector)

Return an array of 2 word Tuples of consisting of adjacent words in the given array 'text'.
"""
function bigrams(text::AbstractVector)
    return collect(Tuple((text[i:(i + 1)]...,)) for i in 1:(length(text) - 1));
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
        pushfirst!(sequence, words[idx]);
        idx = idx - length(words[idx]);
    end
    return sequence, best[end];
end

#=

    DocumentMetadata is the metadata for a document.

=#
struct DocumentMetadata
    title::String
    url::String
    number_of_words::Int64

    function DocumentMetadata(title::String, url::String, number_of_words::Int64)
        return new(title, url, number_of_words);
    end
end

abstract type AbstractInformationRetrievalSystem end;

#=

    InformationRetrievalSystem is a information retrieval (IR) system implementation

    that consists of an index, a set of stop words, and the metadata for the documents.

=#
struct InformationRetrievalSystem <: AbstractInformationRetrievalSystem
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
    index_document(irs::T, text::String, url::String) where {T <: AbstractInformationRetrievalSystem}

Index the document by its text 'text' and URL 'url'.
"""
function index_document(irs::T, text::String, url::String) where {T <: AbstractInformationRetrievalSystem}
    local title::String = strip(text[1:(findfirst("\n", text).stop)]);
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
    index_collection(irs::T, filenames::AbstractVector) where {T <: AbstractInformationRetrievalSystem}

Index the given collection of files 'filenames'.
"""
function index_collection(irs::T, filenames::AbstractVector) where {T <: AbstractInformationRetrievalSystem}
    for filename in filenames
        index_document(irs, String(read(filename)), relpath(filename, AIMAJULIA_DIRECTORY));
    end
    return nothing;
end

"""
    score_document(irs::T, word::String, document_id::Int64) where {T <: AbstractInformationRetrievalSystem}

Return a score for the given word 'word' and document referenced by ID 'document_id'.
"""
function score_document(irs::T, word::String, document_id::Int64) where {T <: AbstractInformationRetrievalSystem}
    return (log(1 + get!(get!(irs.index, word, Dict()), document_id, 0)) / log(1 + irs.documents[document_id].number_of_words));
end

"""
    total_score_document(irs::T, words::AbstractVector, document_id::Int64) where {T <: AbstractInformationRetrievalSystem}

Return the sum of scores for the given words 'words' within the document referenced by ID 'document_id'.
"""
function total_score_document(irs::T, words::AbstractVector, document_id::Int64) where {T <: AbstractInformationRetrievalSystem}
    return sum(score_document(irs, word, document_id) for word in words);
end

"""
    execute_query(irs::T, query::String; n::Int64=10) where {T <: AbstractInformationRetrievalSystem}

Return an array of at most 'n' best matches (score, document ID) Tuples for the given
query string 'query' and IR system 'irs'.

If the query starts with 'learn: ', the following command within the query is executed.
Then the command output is then indexed with index_document().
"""
function execute_query(irs::T, query::String; n::Int64=10) where {T <: AbstractInformationRetrievalSystem}
    if (startswith(query, "learn:"))
        local truncated_query::String = strip(query[7:end]);
        local document_text::String = strip(read(pipeline(stdin, `$truncated_query`), String));
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
struct UnixConsultant <: AbstractInformationRetrievalSystem
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

function all_shift_ciphers(text::String)
    return collect(shift_encode(text, i) for (i, letter) in enumerate(lowercase_alphabet));
end

#=

    ShiftCipherDecoder contains the probability distribution for the bigrams of the

    given training text. The decoder tries all 26 possible encodings and returns

    the highest scoring decoded text.

=#
struct ShiftCipherDecoder
    training_text::String
    P2::CountingProbabilityDistribution

    function ShiftCipherDecoder(training_text::String)
        local canonicalized_text::String = canonicalize_text(training_text)
        return new(canonicalized_text, CountingProbabilityDistribution(bigrams(canonicalized_text), default=1));
    end
end

"""
    score_text(scd::ShiftCipherDecoder, plaintext::String)

Return a score for the given text 'plaintext' by using the probability distribution 'scd.P2' for letter pairs.
"""
function score_text(scd::ShiftCipherDecoder, plaintext::String)
    local score::Float64 = 1.0;
    for bigram in bigrams(plaintext)
        score = score * scd.P2[bigram];
    end
    return score;
end

"""
    decode_text(scd::ShiftCipherDecoder, ciphertext::String)

Return the decoded ciphertext using the best scoring cipher.
"""
function decode_text(scd::ShiftCipherDecoder, ciphertext::String)
    return argmax(all_shift_ciphers(ciphertext),
                    (function(shifted_text::String)
                        return score_text(scd, shifted_text);
                    end));
end

#=

    PermutationCipherDecoder contains the probability distribution for the words of the training text, the

    probability distribution for the 1-grams (letters) of the training text, and the probability distribution

    for the 2-grams (2 adjacent letters) of the training text.



    This decoder does not try all possible encodings because there are 26! permutations. As a result, the

    decoder tries to search for a solution. The decoder would have some success by simply guessing by with

    only the 1-grams of letters, but, this decoder uses the incremental representation. Each state is an 

    array of letter to letter mappings (ie. ('z', 'e') represents that the letter 'z' will translate to 'e').

=#
mutable struct PermutationCipherDecoder
    P_words::UnigramWordModel
    P1::UnigramWordModel
    P2::NgramWordModel
    character_domain::Set
    ciphertext::String

    function PermutationCipherDecoder(training_text::String)
        return new(UnigramWordModel(extract_words(training_text)),
                    UnigramWordModel(collect(training_text)),
                    NgramWordModel(2, extract_words(training_text)));
    end
end

"""
    score_text(pcd::PermutationCipherDecoder, code::AbstractVector)

Return a score for the given code 'code' by obtaining the product of word scores, 1-gram scores,
and 2-gram scores. Since these values can get very small, this function will use the logarithms
of the scores to calculate the result.
"""
function score_text(pcd::PermutationCipherDecoder, code::AbstractVector)
    local full_code::Dict = Dict(code);
    local new_characters::Dict = Dict(collect((x, x)
                                                for x in pcd.character_domain
                                                if (!(haskey(full_code, x)))));
    merge!(full_code, new_characters);
    full_code[' '] = ' ';
    local text::String = String(map((function(c::Char)
                                        return full_code[c];
                                    end),
                                    pcd.ciphertext));
    local log_P::Float64 = (sum((log(pcd.P_words[word]) + 1e-20) for word in extract_words(text)) +
                            sum((log(pcd.P1[c] + 0.00001) for c in text)) +
                            sum((log(pcd.P2[bigram] + 1e-10) for bigram in bigrams(text))));
    return -exp(log_P);
end

"""
    decode_text(pcd::PermutationCipherDecoder, ciphertext::String)

Return the decoded ciphertext by searching for a decoding of the given ciphertext 'ciphertext'.
"""
function decode_text(pcd::PermutationCipherDecoder, ciphertext::String)
    pcd.ciphertext = ciphertext;
    pcd.character_domain = Set(collect(c for c in ciphertext
                                        if (c != ' ')));
    local problem::PermutationCipherDecoderProblem = PermutationCipherDecoderProblem(pcd);
    local solution::Node = best_first_graph_search(problem,
                                                    (function(node::Node)
                                                        return score_text(pcd, node.state);
                                                    end));
    local solution_dict::Dict = Dict(solution.state);
    solution_dict[' '] = ' ';
    return String(map((function(c::Char)
                            return get(solution_dict, c, c);
                        end),
                        pcd.ciphertext));
end

#=

    PermutationCipherDecoderProblem is the problem of decoding a ciphertext when there

    are 26! possible encoding permutations.

=#
struct PermutationCipherDecoderProblem <: AbstractProblem
    initial::AbstractVector
    decoder::PermutationCipherDecoder

    function PermutationCipherDecoderProblem(decoder::PermutationCipherDecoder; initial::Union{Nothing, AbstractVector}=nothing)
        if (typeof(initial) <: Nothing)
            return new([], decoder);
        else
            return new(initial, decoder);
        end
    end
end

"""
    actions(pcdp::PermutationCipherDecoderProblem, state::AbstractVector)

Return an array of possible actions that can be executed in the given state 'state'.
"""
function actions(pcdp::PermutationCipherDecoderProblem, state::AbstractVector)
    local state_dict::Dict = Dict(state);
    local search_list::AbstractVector = collect(character
                                                for character in pcdp.decoder.character_domain
                                                if (!(haskey(state_dict, character))));
    local target_list::AbstractVector = collect(character
                                                for character in lowercase_alphabet
                                                if (!(character in values(state_dict))));
    local plain_character::Char = argmax(search_list,
                                        (function(c::Char)
                                            return pcdp.decoder.P1[c];
                                        end));
    return collect(zip(Base.Iterators.repeated(plain_character, length(target_list)), target_list));
end

"""
    get_result(pcdp::PermutationCipherDecoderProblem, state::AbstractVector, action::Tuple)

Return the resulting state from executing the given action 'action' in the given state 'state'.
"""
function get_result(pcdp::PermutationCipherDecoderProblem, state::AbstractVector, action::Tuple)
    local new_state::Dict = Dict(state);
    new_state[action[1]] = action[2];

    # All states for the problem 'pcdp' should be in sorted order.
    return sort(collect((x, y) for (x, y) in new_state),
                lt=(function(t1::Tuple, t2::Tuple)
                        return t1[1] < t2[1];
                    end));
end

"""
    goal_test(pcdp::PermutationCipherDecoderProblem, state::AbstractVector)

Return a boolean value representing whether all characters in the character domain have a corresponding
mapping in the given state 'state'.
"""
function goal_test(pcdp::PermutationCipherDecoderProblem, state::AbstractVector)
    return (length(state) >= length(pcdp.decoder.character_domain));
end

"""
    path_cost(pcdp::PermutationCipherDecoderProblem, cost::Float64, state_1::AbstractVector, action::Tuple, state_2::AbstractVector)

Return the cost of a solution path arriving at 'state_2' from 'state_1' with the given action 'action' and
cost 'cost' to arrive at 'state_1'.
"""
function path_cost(pcdp::PermutationCipherDecoderProblem, cost::Float64, state_1::AbstractVector, action::Tuple, state_2::AbstractVector)
    return cost + 1;
end

