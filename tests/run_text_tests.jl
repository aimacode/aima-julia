include("../aimajulia.jl");

using Base.Test;

using aimajulia;

using aimajulia.utils;

#The following text tests are from the aima-python doctests

flatland = String(read("./aima-data/EN-text/flatland.txt"));
word_sequence = extract_words(flatland);
P1 = UnigramWordModel(word_sequence);
P2 = NgramWordModel(2, word_sequence);
P3 = NgramWordModel(3, word_sequence);

@test (top(P1, 5) == [(2081,"the"),(1479,"of"),(1021,"and"),(1008,"to"),(850,"a")]);

@test (top(P2, 5) == [(368,("of","the")),(152,("to","the")),(152,("in","the")),(86,("of","a")),(80,("it","is"))]);

@test (top(P3, 5) == [(30,("a","straight","line")),
                        (19,("of","three","dimensions")),
                        (16,("the","sense","of")),
                        (13,("by","the","sense")),
                        (13,("as","well","as"))]);

@test (abs(P1["the"] - 0.0611) <= (0.001 * max(abs(P1["the"]), abs(0.0611))));

@test (abs(P2[("of", "the")] - 0.0108) <= (0.01 * max(abs(P2[("of", "the")]), abs(0.0108))));

@test (abs(P3[("so", "as", "to")] - 0.000323) <= (0.001 * max(abs(P3[("so", "as", "to")]), abs(0.000323))));

@test (!haskey(P2.conditional_probabilities, ("went",)));

@test (P3.conditional_probabilities[("in", "order")].dict == Dict([Pair("to", 6)]));

test_string = "unigram";
word_sequence = extract_words(test_string);
P1 = UnigramWordModel(word_sequence);

@test (P1.dict == Dict([Pair("unigram", 1)]));

test_string = "bigram text";
word_sequence = extract_words(test_string);
P2 = NgramWordModel(2, word_sequence);

@test (P2.dict == Dict([Pair(("bigram", "text"), 1)]));

test_string = "test trigram text here";
word_sequence = extract_words(test_string);
P3 = NgramWordModel(3, word_sequence);

@test (haskey(P3.dict, ("test", "trigram", "text")));

@test (haskey(P3.dict, ("trigram", "text", "here")));

@test (extract_words("``EGAD'' Edgar cried.") == ["egad", "edgar", "cried"]);

@test (canonicalize_text("``EGAD'' Edgar cried.") == "egad edgar cried");

