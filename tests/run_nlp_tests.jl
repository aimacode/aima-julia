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

grammar = aimajulia.nlp.epsilon_chomsky;

@test (all((function(rule::Tuple)
                return (length(rule) == 3);
            end), cnf_rules(grammar)));

lexicon = Lexicon(["Article"=>"the | a | an", "Pronoun"=>"i | you | he"]);
rules = Rules(["S"=>"Article | More | Pronoun", "More"=>"Article Pronoun | Pronoun Pronoun"]);
grammar = Grammar("Simplegram", rules, lexicon);
sentence = generate_random_sentence(grammar, "S");

@test (all((function(token::String)
                return any((function(terminals::AbstractVector)
                                return (token in terminals);
                            end), values(grammar.lexicon));
            end), map(String, split(sentence))));

check = Dict([Pair("A", [(["B", "C"], 0.3), (["D", "E"], 0.7)]),
            Pair("B", [(["E"], 0.1), (["a"], 0.2), (["b", "c"], 0.7)])]);

@test (ProbabilityRules(["A"=>"B C [0.3] | D E [0.7]", "B"=>"E [0.1] | a [0.2] | b c [0.7]"]) == check);

check = Dict([Pair("Article", [("the", 0.5), ("a", 0.25), ("an", 0.25)]),
            Pair("Pronoun", [("i", 0.4), ("you", 0.3), ("he", 0.3)])]);

@test (ProbabilityLexicon(["Article"=>"the [0.5] | a [0.25] | an [0.25]", "Pronoun"=>"i [0.4] | you [0.3] | he [0.3]"]) == check);

rules = ProbabilityRules(["A"=>"B C [0.3] | D E [0.7]", "B"=>"E [0.1] | a [0.2] | b c [0.7]"]);
lexicon = ProbabilityLexicon(["Article"=>"the [0.5] | a [0.25] | an [0.25]", "Pronoun"=>"i [0.4] | you [0.3] | he [0.3]"]);
grammar = ProbabilityGrammar("Simplegram", rules, lexicon);

@test (rewrites_for(grammar, "A") == [(["B", "C"], 0.3), (["D", "E"], 0.7)]);

@test (is_category(grammar, "the", "Article") == true);

grammar = aimajulia.nlp.epsilon_probability_chomsky;

@test (all((function(rule::Tuple)
                return (length(rule) == 4);
            end), cnf_rules(grammar)));

lexicon = ProbabilityLexicon(["Verb"=>"am [0.5] | are [0.25] | is [0.25]",
                            "Pronoun"=>"i [0.4] | you [0.3] | he [0.3]"]);
rules = ProbabilityRules(["S"=>"Verb [0.5] | More [0.3] | Pronoun [0.1] | nobody is here [0.1]",
                        "More"=>"Pronoun Verb [0.7] | Pronoun Pronoun [0.3]"]);
grammar = ProbabilityGrammar("Simplegram", rules, lexicon);
sentence = generate_random_sentence(grammar, "S");

@test (length(sentence) == 2);

chart = Chart(aimajulia.nlp.epsilon_0);

@test (length(parse_sentence(chart, "the stench is in 2 2")) == 1);

grammar = aimajulia.nlp.epsilon_probability_chomsky;
words = ["the", "robot", "is", "good"];

@test (length(cyk_parse(words, grammar)) == 52);

