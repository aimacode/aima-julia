include("../aimajulia.jl");

using Test;

using Main.aimajulia;

using Main.aimajulia.utils;

#The following nlp tests are from the aima-python doctests

# Test the Grammar DataType methods
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

# Test the ProbabilityGrammar DataType methods
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

# Test the Chart DataType
chart = Chart(aimajulia.nlp.epsilon_0);

@test (length(parse_sentence(chart, "the stench is in 2 2")) == 1);

# Test the CYK parsing
grammar = aimajulia.nlp.epsilon_probability_chomsky;
words = ["the", "robot", "is", "good"];

@test (length(cyk_parse(words, grammar)) == 52);

# Test the HTML parsing functions
address = "https://en.wikipedia.org/wiki/Ethics";

page = load_page_html([address]);

page_html = page[address];

@test ((!occursin("<head>", page_html)) && (!occursin("</head>", page_html)));

test_html_1 = ("Keyword String 1: A man is a male human."
                *"Keyword String 2: Like most other male mammals, a man inherits an"
                *"X from his mom and a Y from his dad."
                *"Links:"
                *"href=\"https://google.com.au\""
                *"<href=\"/wiki/TestThing\"> href=\"/wiki/TestBoy\""
                *"href=\"/wiki/TestLiving\" href=\"/wiki/TestMan\">");
test_html_2 = "a mom and a dad";
test_html_3 = "<!DOCTYPE html><html><head><title>Page Title</title></head><body><p>AIMA book</p></body></html>";
no_head_test_html_3 = replace(test_html_3, @r_str("<head>(.*)</head>", "s")=>"");

@test ((!occursin("<head>", no_head_test_html_3)) && (!occursin("</head>", no_head_test_html_3)));

@test ((occursin("AIMA book", test_html_3)) && (occursin("AIMA book", no_head_test_html_3)));

page_A = Page("A", inlinks=["B", "C", "E"], outlinks=["D"], hub=1, authority=6);
page_B = Page("B", inlinks=["E"], outlinks=["A", "C", "D"], hub=2, authority=5);
page_C = Page("C", inlinks=["B", "E"], outlinks=["A", "D"], hub=3, authority=4);
page_D = Page("D", inlinks=["A", "B", "C", "E"], outlinks=[], hub=4, authority=3);
page_E = Page("E", inlinks=[], outlinks=["A", "B", "C", "D", "F"], hub=5, authority=2);
page_F = Page("F", inlinks=["E"], outlinks=[], hub=6, authority=1);

page_dict = Dict([Pair(page_A.address, page_A),
                    Pair(page_B.address, page_B),
                    Pair(page_C.address, page_C),
                    Pair(page_D.address, page_D),
                    Pair(page_E.address, page_E),
                    Pair(page_F.address, page_F)]);

pages_index = page_dict;

pages_content = Dict([Pair(page_A.address, test_html_1),
                        Pair(page_B.address, test_html_2),
                        Pair(page_C.address, test_html_1),
                        Pair(page_D.address, test_html_2),
                        Pair(page_E.address, test_html_1),
                        Pair(page_F.address, test_html_2)]);

@test (Set(determine_inlinks(page_A, pages_index)) == Set(["B", "C", "E"]));

@test (Set(determine_inlinks(page_E, pages_index)) == Set([]));

@test (Set(determine_inlinks(page_F, pages_index)) == Set(["E"]));

test_page_A = page_dict[page_A.address];
test_outlinks = find_outlinks(test_page_A, pages_content, only_wikipedia_urls);

@test ("https://en.wikipedia.org/wiki/TestThing" in test_outlinks);

@test (!("https://google.com.au" in test_outlinks));

pages = Dict(collect((k, page_dict[k]) for k in ("F",)));
pages_two = Dict(collect((k, page_dict[k]) for k in ("A", "E")));
expanded_pages = expand_pages(pages, pages_index);

@test (all(x in keys(expanded_pages) for x in ("F", "E")));

@test (all(!(x in keys(expanded_pages)) for x in ("A", "B", "C", "D")));

expanded_pages = expand_pages(pages_two, pages_index);

@test (all(x in keys(expanded_pages) for x in ("A", "B", "C", "D", "E", "F")));

pages = relevant_pages("his dad", pages_index, pages_content);

@test (all((x in keys(pages)) for x in ("A", "C", "E")));

@test (all((!(x in keys(pages))) for x in ("B", "D", "F")));

pages = relevant_pages("mom and dad", pages_index, pages_content);

@test (all((x in keys(pages)) for x in ("A", "B", "C", "D", "E", "F")));

pages = relevant_pages("philosophy", pages_index, pages_content);

@test (all((!(x in keys(pages))) for x in ("A", "B", "C", "D", "E", "F")));

normalize_pages(page_dict);
println("pages_index hubs: ", collect(page.hub for page in values(pages_index)));

expected_hubs = [(1 / sqrt(91)), (2 / sqrt(91)), (3 / sqrt(91)), (4 / sqrt(91)), (5 / sqrt(91)), (6 / sqrt(91))];
expected_authorities = collect(reverse(expected_hubs));

@test (length(expected_hubs) == length(expected_authorities) == length(pages_index));

sorted_pages = sort(collect(pages_index),
                    lt=(function(p1::Pair, p2::Pair)
                            if (p1.first < p2.first)
                                return true;
                            else
                                return false;
                            end
                        end));

@test (expected_hubs == collect(page.hub for (address, page) in sorted_pages));

@test (expected_authorities == collect(page.authority for (address, page) in sorted_pages));

convergence = ConvergenceDetector();
detect_convergence(convergence, pages_index);

@test (detect_convergence(convergence, pages_index));

new_pages_index = deepcopy(pages_index);

for (address, page) in new_pages_index
    page.hub = page.hub + 0.0003;
    page.authority = page.authority + 0.0004;
end

@test (detect_convergence(convergence, new_pages_index));

for (address, page) in new_pages_index
    page.hub = page.hub + 3000000;
    page.authority = page.authority + 3000000;
end

@test (!detect_convergence(convergence, new_pages_index));

@test (sort(get_inlinks(page_dict["A"], pages_index)) == page_dict["A"].inlinks);

@test (sort(get_outlinks(page_dict["A"], pages_index, pages_content)) == page_dict["A"].outlinks);

HITS("inherit", pages_index, pages_content);

authorities = [page_A.authority, page_B.authority, page_C.authority, page_D.authority, page_E.authority, page_F.authority];
hubs = [page_A.hub, page_B.hub, page_C.hub, page_D.hub, page_E.hub, page_F.hub];

@test (reduce(max, authorities) == page_D.authority);

@test (reduce(max, hubs) == page_E.hub);

