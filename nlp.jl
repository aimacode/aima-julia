
export Rules, Lexicon, Grammar,
        rewrites_for, is_category, cnf_rules, rewrite, generate_random_sentence,
        ProbabilityRules, ProbabilityLexicon, ProbabilityGrammar,
        Chart, parse_sentence, cyk_parse;

"""
    Rules{T <: Pair}(rules_array::Array{T})

Return a Dict of mappings for symbols (lexical categories) to alternative sequences.
"""
function Rules{T <: Pair}(rules_array::Array{T, 1})
    local rules::Dict = Dict();
    for (lhs, rhs) in rules_array
        rules[lhs] = collect(map(String, split(strip(ss))) for ss in map(String, split(rhs, ['|'])));
    end
    return rules;
end

"""
    Lexicon{T <: Pair}(rules_array::Array{T})

Return a Dict of mappings for symbols (lexical categories) to alternative words.

The lexicon is the list of allowable words.
"""
function Lexicon{T <: Pair}(rules_array::Array{T, 1})
    local rules::Dict = Dict();
    for (lhs, rhs) in rules_array
        rules[lhs] = collect(strip(ss) for ss in map(String, split(rhs, "|")));
    end
    return rules;
end

type Grammar
    name::String
    rules::Dict
    lexicon::Dict
    categories::Dict

    function Grammar(name::String, rules::Dict, lexicon::Dict)
        local ng::Grammar = new(name, rules, lexicon, Dict());
        for (category, words) in ng.lexicon
            for word in words
                ng.categories[word] = push!(get!(ng.categories, word, Array{String, 1}()), category);
            end
        end
        return ng;
    end
end

function rewrites_for(g::Grammar, cat::String)
    return get(g.rules, cat, Array{String, 1}());
end

function is_category(g::Grammar, word::String, cat::String)
    return (cat in g.categories[word]);
end

function cnf_rules(g::Grammar)
    local cnf::AbstractVector = [];
    for (x, rules) in g.rules
        for (y, z) in rules
            push!(cnf, (x, y, z));
        end
    end
    return cnf;
end

function rewrite(g::Grammar, tokens::AbstractVector, into::AbstractVector)
    for token in tokens
        if (token in keys(g.rules))
            rewrite(g, rand(RandomDeviceInstance, g.rules[token]), into);
        elseif (token in keys(g.lexicon))
            push!(into, rand(RandomDeviceInstance, g.lexicon[token]));
        else
            push!(into, token);
        end
    end
    return into;
end

function generate_random_sentence(g::Grammar, categories::String)
    return join(rewrite(g, split(categories), []), " ");
end

function generate_random_sentence(g::Grammar)
    return generate_random_sentence(g, "S");
end

function ProbabilityRules{T <: Pair}(rules_array::Array{T, 1})
    local rules::Dict = Dict();
    for (lhs, rhs) in rules_array
        rules[lhs] = [];
        local rhs_split::AbstractVector = collect(map(String, split(strip(ss))) for ss in map(String, split(rhs, ['|'])));
        for rule in rhs_split
            local rule_probability::Float64 = parse(Float64, rule[end][2:(end - 1)]);
            local rule_tuple::Tuple = (rule[1:(end - 1)], rule_probability);
            push!(rules[lhs], rule_tuple);
        end
    end
    return rules;
end

function ProbabilityLexicon{T <: Pair}(rules_array::Array{T, 1})
    local rules::Dict = Dict();
    for (lhs, rhs) in rules_array
        rules[lhs] = [];
        local rhs_split::AbstractVector = collect(map(String, split(strip(word))) for word in map(String, split(rhs, ['|'])));
        for rule in rhs_split
            local word_probability::Float64 = parse(Float64, rule[end][2:(end - 1)]);
            local rule_word::String = rule[1];
            local rule_tuple::Tuple = (rule_word, word_probability);
            push!(rules[lhs], rule_tuple);
        end
    end
    return rules;
end

type ProbabilityGrammar
    name::String
    rules::Dict
    lexicon::Dict
    categories::Dict

    function ProbabilityGrammar(name::String, rules::Dict, lexicon::Dict)
        local npg::ProbabilityGrammar = new(name, rules, lexicon, Dict());
        for (category, words) in npg.lexicon
            for (word, p) in words
                npg.categories[word] = push!(get!(npg.categories, word, []), (category, p));
            end
        end
        return npg;
    end
end

function rewrites_for(pg::ProbabilityGrammar, cat::String)
    return get(pg.rules, cat, []);
end

function is_category(pg::ProbabilityGrammar, word::String, cat::String)
    return (cat in map(first, pg.categories[word]));
end

function cnf_rules(pg::ProbabilityGrammar)
    local cnf::AbstractVector = [];
    for (x, rules) in pg.rules
        for ((y, z), p) in rules
            push!(cnf, (x, y, z, p));
        end
    end
    return cnf;
end

function rewrite(pg::ProbabilityGrammar, tokens::AbstractVector, into::AbstractVector)
    local p::Float64;

    for token in tokens
        if (token in keys(pg.rules))
            local non_terminal_symbols::AbstractVector;
            non_terminal_symbols, p = weighted_choice(pg.rules[token]);
            into[2] = into[2] * p;
            rewrite(pg, non_terminal_symbols, into);
        elseif (token in keys(pg.lexicon))
            local terminal_symbol::String;
            terminal_symbol, p = weighted_choice(pg.lexicon[token]);
            push!(into[1], terminal_symbol);
            into[2] = into[2] * p;
        else
            push!(into[1], token);
        end
    end
    return into;
end

function generate_random_sentence(pg::ProbabilityGrammar, categories::String)
    local rewritten_sentence::AbstractVector;
    local p::Float64;
    rewritten_sentence, p = rewrite(pg, split(categories), [[], 1.0]);
    return (join(rewritten_sentence, " "), p);
end

function generate_random_sentence(pg::ProbabilityGrammar)
    return generate_random_sentence(pg, "S");
end

# Create a submodule to contain the following example grammars.
# These grammars are defined in the module path "aimajulia.nlp".

module nlp

using aimajulia;

epsilon_0 = Grammar("ε_0",
                    Rules(      # Grammar for ε_0 (Fig. 22.4 2nd edition)
                        ["S"=>"NP VP | S Conjunction S",
                        "NP"=>"Pronoun | Name | Noun | Article Noun | Digit Digit | NP PP | NP RelClause",
                        "VP"=>"Verb | VP NP | VP Adjective | VP PP | VP Adverb",
                        "PP"=>"Preposition NP",
                        "RelClause"=>"That VP"]),
                    Lexicon(    # Lexicon for ε_0 (Fig. 22.3 2nd edition)
                        ["Noun"=>"stench | breeze | glitter | nothing | wumpus | pit | pits | gold | east",
                        "Verb"=>"is | see | smell | shoot | fell | stinks | go | grab | carry | kill | turn | feel",
                        "Adjective"=>"right | left | east | south | back | smelly",
                        "Adverb"=>"here | there | nearby | ahead | right | left | east | south | back",
                        "Pronoun"=>"me | you | I | it",
                        "Name"=>"John | Mary | Boston | Aristotle",
                        "Article"=>"the | a | an",
                        "Preposition"=>"to | in | on | near",
                        "Conjunction"=>"and | or | but",
                        "Digit"=>"0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9"
                        ]));

# An example grammar for testing
epsilon_ = Grammar("ε_",
                    Rules(["S"=>"NP VP",
                            "NP"=>"Art N | Pronoun",
                            "VP"=>"V NP"]),
                    Lexicon(["Art"=>"the | a",
                            "N"=>"man | woman | table | shoelace | saw",
                            "Pronoun"=>"I | you | it",
                            "V"=>"saw | liked | feel"]));

# An example grammar for testing
epsilon_np_ = Grammar("ε_NP_",
                    Rules(["NP"=>"Adj NP | N"]),
                    Lexicon(["Adj"=>"happy | handsome | hairy", "N"=>"man"]));

# ε_probability is a probabilistic grammar found in the Python notebook.
# The rules (Fig 23.2 3rd edition) for the language 'ε_probability' use the probabilities from the Python notebook.
epsilon_probability = ProbabilityGrammar("ε_probability", 
                                        ProbabilityRules(["S"=>"NP VP [0.6] | S Conjunction S [0.4]",
                                                        "NP"=>("Pronoun [0.2] | Name [0.05] | Noun [0.2] | "*
                                                                "Article Noun [0.15] | Article Adjs Noun [0.1] | "*
                                                                "Digit [0.05] | NP PP [0.15] | NP RelClause [0.1]"),
                                                        "VP"=>("Verb [0.3] | VP NP [0.2] | VP Adjective [0.25] | "*
                                                                "VP PP [0.15] | VP Adverb [0.1]"),
                                                        "Adjs"=>"Adjective [0.5] | Adjective Adjs [0.5]",
                                                        "PP"=>"Preposition NP [1]",
                                                        "RelClause"=>"RelPro VP [1]"]),
                                        ProbabilityLexicon(["Verb"=>"is [0.5] | say [0.3] | are [0.2]",
                                                            "Noun"=>"robot [0.4] | sheep [0.4] | fence [0.2]",
                                                            "Adjective"=>"good [0.5] | new [0.2] | sad [0.3]",
                                                            "Adverb"=>"here [0.6] | lightly [0.1] | now [0.3]",
                                                            "Pronoun"=>"me [0.3] | you [0.4] | he [0.3]",
                                                            "RelPro"=>"that [0.5] | who [0.3] | which [0.2]",
                                                            "Name"=>"john [0.4] | mary [0.4] | peter [0.2]",
                                                            "Article"=>"the [0.5] | a [0.25] | an [0.25]",
                                                            "Preposition"=>"to [0.4] | in [0.3] | at [0.3]",
                                                            "Conjuction"=>"and [0.5] | or [0.2] | but [0.3]",
                                                            "Digit"=>"0 [0.35] | 1 [0.35] | 2 [0.3]"])); 

epsilon_chomsky = Grammar("ε_chomsky",
                        Rules(["S"=>"NP VP",
                                "NP"=>"Article Noun | Adjective Noun",
                                "VP"=>"Verb NP | Verb Adjective"]),
                        Lexicon(["Article"=>"the | a | an",
                                "Noun"=>"robot | sheep | fence",
                                "Adjective"=>"good | new | sad",
                                "Verb"=>"is | say | are"]));

epsilon_probability_chomsky = ProbabilityGrammar("ε_probability_chomsky",
                                                ProbabilityRules(["S"=>"NP VP [1]",
                                                                "NP"=>"Article Noun [0.6] | Adjective Noun [0.4]",
                                                                "VP"=>"Verb NP [0.5] | Verb Adjective [0.5]"]),
                                                ProbabilityLexicon(["Article"=>"the [0.5] | a [0.25] | an [0.25]",
                                                                    "Noun"=>"robot [0.4] | sheep [0.4] | fence [0.2]",
                                                                    "Adjective"=>"good [0.5] | new [0.2] | sad [0.3]",
                                                                    "Verb"=>"is [0.5] | say [0.3] | are [0.2]"]));

end

#=

    Chart is a data structure used in the process of analyzing a string

    of words to uncover the phrase structure.

=#
type Chart
    trace::Bool
    grammar::Grammar
    chart::AbstractVector

    function Chart(g::Grammar; trace::Bool= false)
        return new(trace, g);
    end
end

function parse_sentence(chart::Chart, words::String, categories::String)
    local words_array::Array{String, 1} = map(String, split(words));
    return parse_sentence(chart, words_array, categories);
end

function parse_sentence(chart::Chart, words::String)
    local words_array::Array{String, 1} = map(String, split(words));
    return parse_sentence(chart, words_array, "S");
end

function parse_sentence(chart::Chart, words::Array{String, 1}, categories::String)
    parse_words(chart, words, categories);
    return collect([i, j, categories, found, []]
                    for (i, j, lhs, found, expects) in chart.chart[length(words)]);
end

function parse_sentence(chart::Chart, words::Array{String, 1})
    return parse_sentence(chart, words, "S");
end

function parse_words(chart::Chart, words::Array{String, 1}, categories::String)
    chart.chart = collect([] for i in 1:(length(words) + 1));
    add_edge(chart, [1, 1, "S_", [], [categories]]);
    for i in 1:length(words)
        scanner(chart, i, words[i]);
    end
    return chart.chart;
end

function parse_words(chart::Chart, words::Array{String, 1})
    return parse_words(chart, words, "S");
end

function add_edge(chart::Chart, edge::AbstractVector)
    local start_index::Int64;
    local end_index::Int64;
    local lhs::String;
    local found::AbstractVector;
    local expects::AbstractVector;

    start_index, end_index, lhs, found, expects = edge;
    if (!(edge in chart.chart[end_index]))
        push!(chart.chart[end_index], edge);
        if (chart.trace)
            println("Chart: added ", edge);
        end
        if (length(expects) == 0)
            extender(chart, edge);
        else
            predictor(chart, edge);
        end
    end
    return nothing;
end

function scanner(chart::Chart, index::Int64, word::String)
    for (i, j, A, alpha, Bb) in chart.chart[index]
        if ((length(Bb) != 0) && (is_category(chart.grammar, word, Bb[1])))
            add_edge(chart, [i, (j + 1), A, vcat(alpha, [(Bb[1], word)]), Bb[2:end]]);
        end
    end
    return nothing;
end

function predictor(chart::Chart, edge::AbstractVector)
    local i::Int64;
    local j::Int64;
    local A::String;
    local alpha::AbstractVector;
    local Bb::AbstractVector;

    i, j, A, alpha, Bb = edge;
    local B::String = Bb[1];
    if (haskey(chart.grammar.rules, B))
        for rhs in rewrites_for(chart.grammar, B)
            add_edge(chart, [j, j, B, [], rhs]);
        end
    end
    return nothing;
end

function extender(chart::Chart, edge::AbstractVector)
    local m::Int64;
    local n::Int64;
    local B::String;
    local found::AbstractVector;
    local expects::AbstractVector;

    m, n, B, found, expects = edge;
    for (i, j, A, alpha, B1b) in chart.chart[m]
        if ((length(B1b) != 0) && (B == B1b[1]))
            add_edge(chart, [i, n, A, vcat(alpha, [edge]), B1b[2:end]]);
        end
    end
    return nothing;
end

"""
    cyk_parse(words::Array{String, 1}, grammar::ProbabilityGrammar)

Return the resulting table as a Dict by applying the CYK algorithm (Fig. 23.5) on the
given sequence of words 'words' and grammar 'grammar'.
"""
function cyk_parse(words::Array{String, 1}, grammar::ProbabilityGrammar)
    local N::Int64 = length(words);
    local P::Dict = Dict();

    for (i, word) in enumerate(words)
        for (X, p) in get!(grammar.categories, word, [])
            P[(X, i, 1)] = p;
        end
    end

    for l in 2:N
        for start_index in 1:(N - l + 1)
            for len1 in 1:(l - 1)
                local len2::Int64 = l - len1;
                for (X, Y, Z, p) in cnf_rules(grammar)
                    P[(X, start_index, l)] = max(get!(P, (X, start_index, l), 0.0),
                                                (get!(P, (Y, start_index, len1), 0.0)
                                                * get!(P, (Z, start_index + len1, len2), 0.0) * p));
                end
            end
        end
    end
    return P;
end


