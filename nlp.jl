
export Rules, Lexicon, Grammar;

"""
	Rules{T <: Pair}(rules_array::Array{T})

Return a Dict of mappings for symbols (lexical categories) to alternative sequences.
"""
function Rules{T <: Pair}(rules_array::Array{T})
	local rules::Dict = Dict();
	for (lhs, rhs) in rules_array
		rules[lhs] = collect(split(strip(ss)) for ss in split(rhs, ['|']));
	end
	return rules;
end

"""
	Lexicon{T <: Pair}(rules_array::Array{T})

Return a Dict of mappings for symbols (lexical categories) to alternative words.

The lexicon is the list of allowable words.
"""
function Lexicon{T <: Pair}(rules_array::Array{T})
	local rules::Dict = Dict();
	for (lhs, rhs) in rules_array
		rules[lhs] = collect(strip(ss) for ss in split(rhs, "|"));
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
        for (symbol, words) in ng.lexicon
            for word in words
                ng.categories[word] = push!(get!(ng.categories, word, Array{String, 1}()), symbol);
            end
        end
        return ng;
    end
end

