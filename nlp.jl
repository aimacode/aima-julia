
export Rules, Lexicon, Grammar,
        rewrites_for, is_category, cnf_rules, rewrite, generate_random_sentence,
        ProbabilityRules, ProbabilityLexicon, ProbabilityGrammar;

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
        for (y, z) in rules;
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
            local rule_tuple::Tuple = (word, word_probability);
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
        local npg::Grammar = new(name, rules, lexicon, Dict());
        for (category, words) in npg.lexicon
            for (word, p) in words
                npg.categories[word] = push!(get!(npg.categories, word, []), (category, p));
            end
        end
        return npg;
    end
end

