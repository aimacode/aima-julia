
export Rules, Lexicon, Grammar,
        rewrites_for, is_category, cnf_rules, rewrite, generate_random_sentence,
        ProbabilityRules, ProbabilityLexicon, ProbabilityGrammar,
        Chart, parse_sentence, cyk_parse,
        Page, load_page_html, determine_inlinks, find_outlinks, init_pages, only_wikipedia_urls,
        expand_pages, relevant_pages, normalize_pages,
        ConvergenceDetector, detect_convergence, get_inlinks, get_outlinks, HITS;

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

#=

    Grammar consists of a set of rules and a lexicon.

=#
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

"""
    rewrites_for(g::Grammar, cat::String)

Return an Array of possible rhs's that category 'cat' can be rewritten as.
"""
function rewrites_for(g::Grammar, cat::String)
    return get(g.rules, cat, Array{String, 1}());
end

"""
    is_category(g::Grammar, word::String, cat::String)

Return whether the given word 'word' is of category 'cat'.
"""
function is_category(g::Grammar, word::String, cat::String)
    return (cat in g.categories[word]);
end

"""
    cnf_rules(g::Grammar)

Return the rules of grammar 'g' as an Array of Tuples (X, Y, Z) such that X -> Y Z.
"""
function cnf_rules(g::Grammar)
    local cnf::AbstractVector = [];
    for (x, rules) in g.rules
        for (y, z) in rules
            push!(cnf, (x, y, z));
        end
    end
    return cnf;
end

"""
    rewrite(g::Grammar, tokens::AbstractVector, into::AbstractVector)

Return the resulting array of words by replacing each token in 'tokens' with a random word.
"""
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

"""
    generate_random_sentence(g::Grammar, categories::String)
    generate_random_sentence(g::Grammar)

Return a randomly generated sentence as a String by using the given categories 'categories'.
"""
function generate_random_sentence(g::Grammar, categories::String)
    return join(rewrite(g, split(categories), []), " ");
end

function generate_random_sentence(g::Grammar)
    return generate_random_sentence(g, "S");
end

"""
    ProbabilityRules{T <: Pair}(rules_array::Array{T})

Return a Dict of mappings for symbols (lexical categories) to alternative sequences with probabilities.
"""
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

"""
    ProbabilityLexicon{T <: Pair}(rules_array::Array{T})

Return a Dict of mappings for symbols (lexical categories) to alternative words with probabilities.

The lexicon is the list of allowable words.
"""
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

#=

    ProbabilityGrammar consists of a set of rules and a lexicon.

=#
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

"""
    rewrites_for(pg::ProbabilityGrammar, cat::String)

Return an Array of possible rhs's that category 'cat' can be rewritten as.
"""
function rewrites_for(pg::ProbabilityGrammar, cat::String)
    return get(pg.rules, cat, []);
end

"""
    is_category(pg::ProbabilityGrammar, word::String, cat::String)

Return whether the given word 'word' is of category 'cat'.
"""
function is_category(pg::ProbabilityGrammar, word::String, cat::String)
    return (cat in map(first, pg.categories[word]));
end

"""
    cnf_rules(pg::ProbabilityGrammar)

Return the rules of grammar 'pg' as an Array of Tuples (X, Y, Z, p) such that X -> Y Z [p].
"""
function cnf_rules(pg::ProbabilityGrammar)
    local cnf::AbstractVector = [];
    for (x, rules) in pg.rules
        for ((y, z), p) in rules
            push!(cnf, (x, y, z, p));
        end
    end
    return cnf;
end

"""
    rewrite(pg::ProbabilityGrammar, tokens::AbstractVector, into::AbstractVector)

Return the resulting array of words by replacing each token in 'tokens' with a random word.
"""
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

"""
    generate_random_sentence(pg::ProbabilityGrammar, categories::String)
    generate_random_sentence(pg::ProbabilityGrammar)

Return a randomly generated sentence as a String by using the given categories 'categories'.
"""
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

"""
    parse_sentence(chart::Chart, words::String, categories::String)
    parse_sentence(chart::Chart, words::String)
    parse_sentence(chart::Chart, words::Array{String, 1}, categories::String)
    parse_sentence(chart::Chart, words::Array{String, 1})

Return an array of parses given the sentence 'words' and categories 'categories'.
"""
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

"""
    parse_words(chart::Chart, words::Array{String, 1}, categories::String)
    parse_words(chart::Chart, words::Array{String, 1})

Return an array of words given the array of 'words' and categories 'categories'.
"""
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

"""
    add_edge(chart::Chart, edge::AbstractVector)

Add the given edge 'edge' to the chart 'chart'.
"""
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

"""
    scanner(chart::Chart, index::Int64, word::String)

Extend the edge for given index 'index' if the given word 'word' and its category are expected.
"""
function scanner(chart::Chart, index::Int64, word::String)
    for (i, j, A, alpha, Bb) in chart.chart[index]
        if ((length(Bb) != 0) && (is_category(chart.grammar, word, Bb[1])))
            add_edge(chart, [i, (j + 1), A, vcat(alpha, [(Bb[1], word)]), Bb[2:end]]);
        end
    end
    return nothing;
end

"""
    predictor(chart::Chart, edge::AbstractVector)

Add edges with rules for 'B' that may help in extending the given edge 'edge'.
"""
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

"""
    extender(chart::Chart, edge::AbstractVector)

Extend whatever edge can be extended from the given edge 'edge' to the chart 'chart'.
"""
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

#=

    Page consists of an address, an array of inlinks, an array of outlinks, a hub score, and an authority score.

=#
type Page
    address::String
    inlinks::AbstractVector
    outlinks::AbstractVector
    hub::Float64
    authority::Float64

    function Page(address::String; inlinks::AbstractVector=[], outlinks::AbstractVector=[], hub::Int64=0, authority::Int64=0)
        return new(address, inlinks, outlinks, hub, authority);
    end
end

"""
    load_page_html(addresses::AbstractVector)

Return a Dict of page content for the given URLs 'addresses' by downloading the HTML pages.
"""
function load_page_html(addresses::AbstractVector)
    local content::Dict = Dict();
    for address in addresses
        local tmp_file::Tuple = mktemp(pwd());
        close(tmp_file[2]);
        rm(tmp_file[1]);
        download(address, tmp_file[1]);
        local raw_html::String = String(read(tmp_file[1]));
        rm(tmp_file[1]);
        local html::String = replace(raw_html, @r_str("<head>(.*)</head>", "s"), "");
        content[address] = html;
    end
    return content;
end

"""
    determine_inlinks(page::Page, pages_index::Dict)

Return an Array of inlinks for the given page 'page'.
"""
function determine_inlinks(page::Page, pages_index::Dict)
    local inlinks::AbstractVector = [];
    for (address, index_page) in pages_index
        if (page.address == index_page.address)
            continue;
        elseif (page.address in index_page.outlinks)
            push!(inlinks, address);
        end
    end
    return inlinks;
end

"""
    find_outlinks(page::Page, pages_content::Dict)
    find_outlinks(page::Page, pages_content::Dict, handle_urls::Function)

Return an Array of outlinks to other pages for the given page 'page'.

If the argument 'handle_urls' is given, the resulting array is returned
after the function is applied to the array of outlinks.
"""
function find_outlinks(page::Page, pages_content::Dict)
    local urls::AbstractVector = Array{String, 1}();
    for regex_m in eachmatch(r"href=['\"]?([^'\" >]+)", pages_content[page.address])
        push!(urls, String(regex_m.captures[1]));
    end
    return urls;
end

function find_outlinks(page::Page, pages_content::Dict, handle_urls::Function)
    local urls::AbstractVector = Array{String, 1}();
    for regex_m in eachmatch(r"href=['\"]?([^'\" >]+)", pages_content[page.address])
        push!(urls, String(regex_m.captures[1]));
    end
    return handle_urls(urls);
end

"""
    init_pages(addresses::AbstractVector)

Return a Dict of pages from the given array of URLs 'addresses'.
"""
function init_pages(addresses::AbstractVector)
    local pages::Dict = Dict();
    for address in addresses
        pages[address] = Page(address);
    end
    return pages;
end

"""
    only_wikipedia_urls(urls::AbstractVector)

Return an Array of wikipedia URLs where relative wiki URLs are converted to their absolute URL.
"""
function only_wikipedia_urls(urls::AbstractVector)
    local wiki_urls::AbstractVector = collect(url for url in urls if (startswith(url, "/wiki/")));
    return collect(("https://en.wikipedia.org" * url) for url in wiki_urls);
end

"""
    expand_pages(pages::Dict, pages_index::Dict)

Return the Dict of expanded pages by adding in every page that links to
or is linked from one of the relevant pages.
"""
function expand_pages(pages::Dict, pages_index::Dict)
    local expanded::Dict = Dict();
    for (address, page) in pages
        if (!haskey(expanded, address))
            expanded[address] = page;
        end
        for inlink in page.inlinks
            if (!haskey(expanded, inlink))
                expanded[inlink] = pages_index[inlink];
            end
        end
        for outlink in page.outlinks
            if (!haskey(expanded, outlink))
                expanded[outlink] = pages_index[outlink];
            end
        end
    end
    return expanded;
end

"""
    relevant_pages(query::String, pages_index::Dict, pages_content::Dict)

Return a Dict of pages that contain all of the query words in 'query'.
These pages are found by intersecting the hit lists of the query words.
"""
function relevant_pages(query::String, pages_index::Dict, pages_content::Dict)
    local hit_intersection::Set = Set(collect(keys(pages_index)));
    local query_words::AbstractVector = map(String, split(query));
    for query_word in query_words
        local hit_list::Set = Set();
        for address in keys(pages_index)
            if (contains(lowercase(pages_content[address]), lowercase(query_word)))
                push!(hit_list, address);
            end
        end
        hit_intersection = intersect(hit_intersection, hit_list);
    end
    return Dict(collect((address, pages_index[address]) for address in hit_intersection));
end

"""
    normalize_pages(pages::Dict)

Divide the scores of each page in 'pages' by the sum of all the squares of all pages' scores
(separately for both the authority and hubs scores). 
"""
function normalize_pages(pages::Dict)
    local summed_hub::Float64 = sum(collect(page.hub^2 for page in values(pages)));
    local summed_authority::Float64 = sum(collect(page.authority^2 for page in values(pages)));
    local sqrt_summed_hub::Float64 = sqrt(summed_hub);
    local sqrt_summed_authority::Float64 = sqrt(summed_authority);
    for address in keys(pages)
        pages[address].hub = pages[address].hub / sqrt_summed_hub;
        pages[address].authority = pages[address].authority / sqrt_summed_authority;
    end
    return nothing;
end

#=

    ConvergenceDetector contains the hub history and authorities history for a set of pages.

=#
type ConvergenceDetector
    hub_history::AbstractVector
    authority_history::AbstractVector

    function ConvergenceDetector()
        return new([], []);
    end
end

"""
    detect_convergence(cd::ConvergenceDetector, pages_index::Dict)

Return a boolean indicating if both the 'hub' and 'authority' values of the pages
in 'pages_index' have converged.
"""
function detect_convergence(cd::ConvergenceDetector, pages_index::Dict)
    local current_hubs::AbstractVector = collect(page.hub for page in values(pages_index));
    local current_authorities::AbstractVector = collect(page.authority for page in values(pages_index));
    if (length(cd.hub_history) != 0)
        local diffs_hub::AbstractVector = collect(abs(x - y) for (x, y) in zip(current_hubs, cd.hub_history[end]));
        local diffs_authority::AbstractVector = collect(abs(x - y) for (x, y) in zip(current_authorities, cd.authority_history[end]));
        local average_delta_hub::Float64 = sum(diffs_hub)/length(pages_index);
        local average_delta_authority::Float64 = sum(diffs_authority)/length(pages_index);
        if ((average_delta_hub < 0.01) && (average_delta_authority < 0.01))
            return true;
        end
    end
    if (length(cd.hub_history) > 2)
        deleteat!(cd.hub_history, 1);
        deleteat!(cd.authority_history, 1);
    end
    push!(cd.hub_history, current_hubs);
    push!(cd.authority_history, current_authorities);
    return false;
end

"""
    get_inlinks(page::Page, pages_index::Dict)

Return an Array of addresses where each address is in both the page's inlinks
and index of pages 'pages_index'.
"""
function get_inlinks(page::Page, pages_index::Dict)
    if (length(page.inlinks) == 0)
        page.inlinks = determine_inlinks(page, pages_index);
    end
    return collect(address for (address, p) in pages_index if (address in page.inlinks));
end

"""
    get_outlinks(page::Page, pages_index::Dict, pages_content::Dict)

Return an Array of addresses where each address is in both the page's outlinks
and index of pages 'pages_index'.
"""
function get_outlinks(page::Page, pages_index::Dict, pages_content::Dict)
    if (length(page.outlinks) == 0)
        page.outlinks = find_outlinks(page, pages_content);
    end
    return collect(address for (address, p) in pages_index if (address in page.outlinks));
end

"""
    HITS(query::String, pages_index::Dict, pages_content::Dict)

Return the computed hubs and authorities with respect to the query as a Dict by using the
HITS algorithm (Fig. 22.1) to the given query 'query', index of pages 'pages_index',
and the content of the pages 'pages_content'.
"""
function HITS(query::String, pages_index::Dict, pages_content::Dict)
    local pages::Dict = expand_pages(relevant_pages(query, pages_index, pages_content), pages_index);
    for p in values(pages)
        p.authority = 1.0;
        p.hub = 1.0;
    end
    local convergence::ConvergenceDetector = ConvergenceDetector();
    while (detect_convergence(convergence, pages_index))
        local authority::Dict = Dict(collect(Pair(p, pages[p].authority) for p in pages));
        local hub::Dict = Dict(collect(Pair(p, pages[p].hub) for p in pages));
        for p in values(pages)
            p.authority = sum(hub[x] for x in get_inlinks(p, pages_index));
            p.hub = sum(authority[x] for x in get_outlinks(p, pages_index, pages_content));
        end
        normalize_pages(pages);
    end
    return pages;
end

