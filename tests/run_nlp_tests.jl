include("../aimajulia.jl");

using Base.Test;

using aimajulia;

using aimajulia.utils;

#The following reinforcement nlp tests are from the aima-python doctests

check = Dict([Pair("A", [["B", "C"], ["D", "E"]]),
            Pair("B", [["E"], ["a"], ["b", "c"]])]);

@test (Rules(["A"=>"B C | D E", "B"=>"E | a | b c"]) == check);

check = Dict([Pair("Article", ["the", "a", "an"]),
            Pair("Pronoun", ["i", "you", "he"])]);

@test (Lexicon(["Article"=>"the | a | an", "Pronoun"=>"i | you | he"]) == check);

rules = Rules(["A"=>"B C | D E", "B"=>"E | a | b c"]);
lexicon = Lexicon(["Article"=>"the | a | an", "Pronoun"=>"i | you | he"]);
grammar = Grammar("Simplegram", rules, lexicon);

@test (rewrites_for(grammar, "A") == [["B", "C"], ["D", "E"]]);

@test (is_category(grammar, "the", "Article") == true);

lexicon = Lexicon(["Article"=>"the | a | an", "Pronoun"=>"i | you | he"]);
rules = Rules(["S"=>"Article | More | Pronoun", "More"=>"Article Pronoun | Pronoun Pronoun"]);
grammar = Grammar("Simplegram", rules, lexicon);
sentence = generate_random_sentence(grammar, "S");

@test (all((function(token::String)
                return any((function(terminals::AbstractVector)
                                return (token in terminals);
                            end), values(grammar.lexicon));
            end), map(String, split(sentence))));

