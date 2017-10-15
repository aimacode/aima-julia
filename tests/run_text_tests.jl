include("../aimajulia.jl");

using Base.Test;

using aimajulia;

using aimajulia.utils;

#The following text tests are from the aima-python doctests

# Base.Test tests for unigram and n-gram of words
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

# Base.Test tests for canonicalizing text
@test (extract_words("``EGAD'' Edgar cried.") == ["egad", "edgar", "cried"]);

@test (canonicalize_text("``EGAD'' Edgar cried.") == "egad edgar cried");

# Base.Test tests for samples() methods
story = String(read("./aima-data/EN-text/flatland.txt"));
story = story*String(read("./aima-data/gutenberg.txt"));
word_sequence = extract_words(story);
P1 = UnigramWordModel(word_sequence);
P2 = NgramWordModel(2, word_sequence);
P3 = NgramWordModel(3, word_sequence);

@test (length(split(samples(P1, 10))) == 10);

@test (length(split(samples(P2, 10))) == 10);

@test (length(split(samples(P3, 10))) == 10);

# Base.Test tests for unigram and n-gram of characters/letters
test_string = "test unigram";
word_sequence = extract_words(test_string);
P1 = UnigramCharModel(word_sequence);
expected_unigrams = Dict([Pair('n',1),
                        Pair('g', 1),
                        Pair('t', 2),
                        Pair('a', 1),
                        Pair('u', 1),
                        Pair('i', 1),
                        Pair('m', 1),
                        Pair('e', 1),
                        Pair('s', 1),
                        Pair('r', 1)]);

@test (length(P1.dict) == length(expected_unigrams));

@test (all(haskey(P1.dict, character) for character in setdiff(Set(test_string), Set(" "))));

test_string = "alpha beta";
word_sequence = extract_words(test_string);
P1 = NgramCharModel(1, word_sequence);

@test (length(P1.dict) == length(Set(collect(test_string))));

@test (all((function(c::Char)
                return haskey(P1.dict, (c,));
            end),
            Set(collect(test_string))));

test_string = "bigram";
word_sequence = extract_words(test_string);
P2 = NgramCharModel(2, word_sequence);
expected_bigrams = Dict([Pair((' ', 'b'), 1),
                        Pair(('b', 'i'), 1),
                        Pair(('i', 'g'), 1),
                        Pair(('g', 'r'), 1),
                        Pair(('r', 'a'), 1),
                        Pair(('a', 'm'), 1)]);

@test (length(P2.dict) == length(expected_bigrams));

@test (all(haskey(P2.dict, key) for key in keys(expected_bigrams)));

@test (all((P2.dict[key] == expected_bigrams[key]) for key in keys(expected_bigrams)));

test_string = "trigram";
word_sequence = extract_words(test_string);
P3 = NgramCharModel(3, word_sequence);
expected_trigrams = Dict([Pair((' ', 't', 'r'), 1),
                        Pair(('t', 'r', 'i'), 1),
                        Pair(('r', 'i', 'g'), 1),
                        Pair(('i', 'g', 'r'), 1),
                        Pair(('g', 'r', 'a'), 1),
                        Pair(('r', 'a', 'm'), 1)]);

@test (length(P3.dict) == length(expected_trigrams));

@test (all(haskey(P3.dict, key) for key in keys(expected_trigrams)));

@test (all((P3.dict[key] == expected_trigrams[key]) for key in keys(expected_trigrams)));

test_string = "trigram trigram trigram";
word_sequence = extract_words(test_string);
P3 = NgramCharModel(3, word_sequence);
expected_trigrams = Dict([Pair((' ', 't', 'r'), 3),
                        Pair(('t', 'r', 'i'), 3),
                        Pair(('r', 'i', 'g'), 3),
                        Pair(('i', 'g', 'r'), 3),
                        Pair(('g', 'r', 'a'), 3),
                        Pair(('r', 'a', 'm'), 3)]);

@test (length(P3.dict) == length(expected_trigrams));

@test (all(haskey(P3.dict, key) for key in keys(expected_trigrams)));

@test (all((P3.dict[key] == expected_trigrams[key]) for key in keys(expected_trigrams)));

# Base.Test tests for encoding
@test (shift_encode("This is a secret message.", 17) == "Kyzj zj r jvtivk dvjjrxv.");

@test (rot13("Hello, world!") == "Uryyb, jbeyq!");

@test (reduce(*, map((function(c::Char)
                            if (c == ' ')
                                return String(['s', ' ', c]);
                            else
                                return String([c]);
                            end
                        end),
                        collect("orange apple lemon "))) == "oranges  apples  lemons  ");

# Base.Test tests for generating arrays of bigrams
@test (bigrams("this") == [('t', 'h'), ('h', 'i'), ('i', 's')]);

@test (bigrams(["this", "is", "a", "test"]) == [("this", "is"), ("is", "a"), ("a", "test")]);

flatland = String(read("./aima-data/EN-text/flatland.txt"));
word_sequence = extract_words(flatland);
P = UnigramWordModel(word_sequence);
segmented_text, p = viterbi_text_segmentation("itiseasytoreadwordswithoutspaces", P);

@test (segmented_text == ["it", "is", "easy", "to", "read", "words", "without", "spaces"]);

