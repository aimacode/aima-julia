
import Base: hash, ==, show;

export hash, ==, show,
        Expression, expr, pretty_set,
        variables, subexpressions, proposition_symbols,
        parse_logic_definite_clause, is_logic_definite_clause, is_logic_symbol,
        is_logic_variable, is_logic_variable_symbol, is_logic_proposition_symbol,
        tt_entails, pl_true, tt_true,
        to_conjunctive_normal_form,
        eliminate_implications, move_not_inwards, distribute_and_over_or,
        associate, dissociate, conjuncts, disjuncts,
        AbstractKnowledgeBase, PropositionalKnowledgeBase, PropositionalDefiniteKnowledgeBase,
        KnowledgeBaseAgentProgram,
        make_percept_sentence, make_action_query, make_action_sentence, execute,
        tell, ask, retract, clauses_with_premise,
        pl_resolution, pl_resolve, pl_fc_entails,
        inspect_literal, unit_clause_assign, find_unit_clause, find_pure_symbol,
        dpll, dpll_satisfiable, walksat, HybridWumpusAgentProgram, plan_route,
        translate_to_sat, extract_solution, sat_plan,
        unify, occurrence_check, extend, substitute,
        standardize_variables, fol_fc_ask,
        FirstOrderLogicKnowledgeBase, fetch_rules_for_goal,
        fol_bc_ask,
        is_number, differentiate, simplify, differentiate_simplify;

abstract AgentProgram;      #declare AgentProgram as a supertype for AgentProgram implementations

immutable Expression
	operator::String
	arguments::Tuple
end

Expression(op::String, args::Vararg{Any}) = Expression(op, map(Expression, args));

function (e::Expression)(args::Vararg{Any})
    if ((length(e.arguments) == 0) && is_logic_symbol(e.operator))
        return Expression(e.operator, map(Expression, args));
    else
        error("ExpressionError: ", e, " is not a Symbol (Nullary Expression)!");
    end
end

# Hashing Expressions and n-Tuple of Expression(s).

hash(e::Expression, h::UInt) = ((hash(e.operator) $ hash(e.arguments)) $ h);

hash(t_e::Tuple{Vararg{Expression}}, h::UInt) = reduce($, vcat(map(hash, collect(t_e)), h));

# Julia does not allow for custom infix operators such as '==>', '<==', '<=>', etc.
# In addition, bitwise xor looks different between Julia ($) and Python (^).

==(e1::Expression, e2::Expression) = ((e1.operator == e2.operator) && (e1.arguments == e2.arguments));

function show(e::Expression)
    if (length(e.arguments) == 0)
        return e.operator;
    elseif (is_logic_symbol(e.operator))
        return @sprintf("%s(%s)", e.operator, join(map(show, map(Expression, e.arguments)), ", "));
    elseif (length(e.arguments) == 1)
        return @sprintf("%s(%s)", e.operator, show(Expression(e.arguments[1])));
    else
        return @sprintf("(%s)", join(map(show, map(Expression, map(string, e.arguments))), @sprintf(" %s ", e.operator)));
    end
end

function show(io::IO, e::Expression)
    print(io, show(e));
    nothing;
end

abstract AbstractKnowledgeBase;

function tell{T <: AbstractKnowledgeBase}(kb::T, e::Expression)
    println("tell() is not implemented yet for ", typeof(kb), "!");
    nothing;
end

function ask{T <: AbstractKnowledgeBase}(kb::T, e::Expression)
    println("ask() is not implemented yet for ", typeof(kb), "!");
    nothing;
end

function retract{T <: AbstractKnowledgeBase}(kb::T, e::Expression)
    println("retract() is not implemented yet for ", typeof(kb), "!");
    nothing;
end

#=

    PropositionalKnowledgeBase is a knowledge base of propositional logic.

=#
type PropositionalKnowledgeBase <: AbstractKnowledgeBase
    clauses::Array{Expression, 1}

    function PropositionalKnowledgeBase()
        return new(Array{Expression, 1}([]));
    end

    function PropositionalKnowledgeBase(e::Expression)
        local pkb = new(Array{Expression, 1}([]));
        tell(pkb, e);
        return pkb;
    end
end

function tell(kb::PropositionalKnowledgeBase, e::Expression)
    append!(kb.clauses, conjuncts(to_conjunctive_normal_form(e)));
    nothing;
end

function ask(kb::PropositionalKnowledgeBase, e::Expression)
    if (tt_entails(Expression("&", kb.clauses...), e))
        return Dict([]);
    else
        return false;
    end
end

function retract(kb::PropositionalKnowledgeBase, e::Expression)
    for conjunct in conjuncts(to_conjunctive_normal_form(e))
        if (conjunct in kb.clauses)
            for (index, item) in enumerate(kb.clauses)
                if (item == conjunct)
                    deleteat!(kb.clauses, index);
                    break;
                end
            end
        end
    end
    nothing;
end

type FirstOrderLogicKnowledgeBase <: AbstractKnowledgeBase
    clauses::Array{Expression, 1}

    function FirstOrderLogicKnowledgeBase()
        return new(Array{Expression, 1}());
    end

    function FirstOrderLogicKnowledgeBase(initial_clauses::Array{Expression, 1})
        local fol_kb = new(Array{Expression, 1}());
        for clause in initial_clauses
            tell(fol_kb, clause);
        end
        return fol_kb;
    end
end

function tell(kb::FirstOrderLogicKnowledgeBase, e::Expression)
    if (is_logic_definite_clause(e))
        push!(kb.clauses, e);
    else
        error("tell(): ", repr(e), " is not a definite clause!");
    end
    nothing;
end

function ask(kb::FirstOrderLogicKnowledgeBase, e::Expression)
    return fol_bc_ask(kb, e);
end

function retract(kb::FirstOrderLogicKnowledgeBase, e::Expression)
    for (index, item) in enumerate(kb.clauses)
        if (item == e)
            deleteat!(kb.clauses, index);
            break;
        end
    end
    nothing;
end

function fetch_rules_for_goal(kb::FirstOrderLogicKnowledgeBase, goal::Expression)
    return kb.clauses;
end

#=

    KnowledgeBaseAgentProgram is a generic knowledge-based implementation of AgentProgram (Fig 7.1).

=#
type KnowledgeBaseAgentProgram <: AgentProgram
    isTracing::Bool
    knowledge_base::AbstractKnowledgeBase
    t::UInt64

    function KnowledgeBaseAgentProgram{T <: AbstractKnowledgeBase}(kb::T; trace::Bool=false)
        return new(trace, kb, UInt64(0));
    end
end

function make_percept_sentence(percept::Expression, t::UInt64)
    return Expression("Percept")(percept, Expression(dec(t)));
end

function make_action_query(t::UInt64)
    return Expression(@sprintf("ShouldDo(action, %s)", dec(t)));
end

function make_action_sentence(action::Dict, t::UInt64)
    return Expression("Did")(action[Expression("action")], Expression(dec(t)));
end

function execute(ap::KnowledgeBaseAgentProgram, percept::Expression)
    tell(ap.knowledge_base, make_percept_sentence(percept, ap.t));
    action = ask(ap.knowledge_base, make_action_query(ap.t));
    tell(ap.knowledge_base, make_action_sentence(action, ap.t));
    ap.t = ap.t + 1;
    if (ap.isTracing)
        @printf("%s perceives %s and does %s\n", string(typeof(ap)), repr(percept), string(action));
    end
    return action;
end


#=

    PropositionalDefiniteKnowledgeBase is a knowledge base of propositional definite clause logic.

=#
type PropositionalDefiniteKnowledgeBase <: AbstractKnowledgeBase
    clauses::Array{Expression, 1}

    function PropositionalDefiniteKnowledgeBase()
        return new(Array{Expression, 1}([]));
    end

    function PropositionalDefiniteKnowledgeBase(e::Expression)
        local pkb = new(Array{Expression, 1}([]));
        tell(pkb, e);
        return pkb;
    end
end

function tell(kb::PropositionalDefiniteKnowledgeBase, e::Expression)
    push!(kb.clauses, e);
    nothing;
end

function ask(kb::PropositionalDefiniteKnowledgeBase, e::Expression)
    if (pl_fc_entails(Expression("&", kb.clauses...), e))
        return Dict([]);
    else
        return false;
    end
end

function retract(kb::PropositionalDefiniteKnowledgeBase, e::Expression)
    for (index, item) in enumerate(kb.clauses)
        if (item == e)
            deleteat!(kb.clauses, index);
            break;
        end
    end
    nothing;
end

function clauses_with_premise(kb::PropositionalDefiniteKnowledgeBase, p::Expression)
    return collect(c for c in kb.clauses if ((c.operator == "==>") && (p in conjuncts(c.arguments[1]))));
end

"""
    pl_fc_entails(kb, q)

Apply the forward-chaining algorithm (Fig. 7.15) for propositional logic.
Return true or false based on whether PropositionalKnowledgeBase 'kb' entails
the proposition symbol 'q'.
"""
function pl_fc_entails(kb::PropositionalDefiniteKnowledgeBase, q::Expression)
    local count::Dict = Dict(collect(Pair(c, length(conjuncts(c.arguments[1]))) for c in kb.clauses if (c.operator == "==>")));
    local agenda::AbstractVector = collect(s for s in kb.clauses if (is_logic_proposition_symbol(s.operator)));
    local inferred::Dict = Dict{Expression, Bool}();
    while (length(agenda) != 0)
        p = shift!(agenda);
        if (p == q)
            return true;
        end
        if (!(get(inferred, p, false)))
            inferred[p] = true;
            for c in clauses_with_premise(kb, p)
                count[c] = count[c] - 1;
                if (count[c] == 0)
                    push!(agenda, c.arguments[2])
                end
            end
        end
    end
    return false;
end

function find_pure_symbol(symbols::Array{Expression, 1}, clauses::Array{Expression, 1})
    for symbol in symbols
        found_positive = false;
        found_negative = false;
        for clause in clauses
            disjuncts_clause = disjuncts(clause);
            if (!found_positive && (symbol in disjuncts_clause))
                found_positive = true;
            end
            if (!found_negative && (Expression("~", symbol) in disjuncts_clause))
                found_negative = true;
            end
        end
        if (found_positive != found_negative)
            return symbol, found_positive;
        end
    end
    return nothing, nothing;
end

function inspect_literal(e::Expression)
    if (e.operator == "~")
        return e.arguments[1], false;
    else
        return e, true;
    end
end

function unit_clause_assign(clause::Expression, model::Dict)
    P = nothing;
    value = nothing;
    for literal in disjuncts(clause)
        symbol, positive = inspect_literal(literal);
        if (haskey(model, symbol))
            if (model[symbol] == positive)
                return nothing, nothing;
            end
        elseif (!(typeof(P) <: Void))
            return nothing, nothing;
        else
            P = symbol;
            value = positive;
        end
    end
    return P, value;
end

function find_unit_clause(clauses::Array{Expression, 1}, model::Dict)
    for clause in clauses
        P, value = unit_clause_assign(clause, model);
        if (!(typeof(P) <: Void))
            return P, value;
        end
    end
    return nothing, nothing;
end

function dpll(clauses::Array{Expression, 1}, symbols::Array{Expression, 1}, model::Dict)
    local unknown_clauses::Array{Expression, 1} = Array{Expression, 1}();
    for clause in clauses
        val = pl_true(clause, model=model);
        if (val == false)
            return false;
        end
        if (val != true)
            push!(unknown_clauses, clause);
        end
    end
    if (length(unknown_clauses) == 0)
        return model;
    end
    P, value = find_pure_symbol(symbols, unknown_clauses);
    if (!(typeof(P) <: Void))
        return dpll(clauses, removeall(symbols, P), extend(model, P, value));
    end
    P, value = find_unit_clause(clauses, model);
    if (!(typeof(P) <: Void))
        return dpll(clauses, removeall(symbols, P), extend(model, P, value));
    end
    P, symbols = symbols[1], symbols[2:end];

    first_recursive_call = dpll(clauses, symbols, extend(model, P, true));
    second_recursive_call = dpll(clauses, symbols, extend(model, P, false));

    if (typeof(first_recursive_call) <: Dict)
        return first_recursive_call;
    elseif (typeof(first_recursive_call) <: Dict)
        return second_recursive_call;
    else
        return false;
    end
end

"""
    dpll_satisfiable(s)

Use the Davis-Putnam-Logemann-Loveland (DPLL) algorithm (Fig. 7.17) to check satisfiability
of the given propositional logic sentence 's' and return the model (dictionary of truth value
assignments) if the sentence 's' is satisfiable and false otherwise.
"""
function dpll_satisfiable(s::Expression)
    local clauses = conjuncts(to_conjunctive_normal_form(s));
    local symbols = proposition_symbols(s);
    return dpll(clauses, symbols, Dict());
end

"""
    walksat(clauses)

Apply the WalkSAT algorithm (Fig. 7.18) to the given 'clauses'. walksat() will return a model
that satisfies the clauses if possible and returns 'nothing' on failure.
"""
function walksat(clauses::Array{Expression, 1}; p::Float64=0.5, max_flips::Int64=10000)
    local symbols::Set{Expression} = Set{Expression}(reduce(vcat,
                                                            collect(collect(symbol for symbol in proposition_symbols(clause))
                                                                    for clause in clauses)));
    local model::Dict{Expression, Bool} = Dict(collect(Pair(symbol, rand(RandomDeviceInstance, [true, false]))
                                                        for symbol in symbols));
    for i in 1:max_flips
        satisfied = Array{Expression, 1}();
        unsatisfied = Array{Expression, 1}();
        for clause in clauses
            push!(if_(pl_true(clause, model=model), satisfied, unsatisfied), clause);
        end
        if (length(unsatisfied) == 0)
            return model;
        end
        clause = rand(RandomDeviceInstance, unsatisfied);
        if (p > rand(RandomDeviceInstance))
            symbol = rand(RandomDeviceInstance, proposition_symbols(clause));
        else
            symbol = argmax(proposition_symbols(clause),
                            (function(e)
                                model[e] = !model[e];
                                count = length((collect(c for c in clauses if (pl_true(c, model=model)))));
                                model[e] = !model[e];
                                return count;
                            end));
        end
        model[symbol] = !model[symbol];
    end
    nothing;
end

#=

    HybridWumpusAgentProgram is a hybrid Wumpus implementation of AgentProgram (Fig. 7.20).

=#
type HybridWumpusAgentProgram <: AgentProgram
    isTracing::Bool
    kb::AbstractKnowledgeBase
    t::UInt64
    plan::AbstractVector

    function HybridWumpusAgentProgram{T <: AbstractKnowledgeBase}(kb::T; trace::Bool=false)
        return new(trace, kb, UInt64(0), []);
    end
end

function plan_route(current, goals, allowed)
    println("plan_route() is not implemented yet!");
    nothing;
end

function execute(ap::HybridWumpusAgentProgram, percept::AbstractVector)
    println("execute() is not implemented yet for HybridWumpusAgentProgram!");
    nothing;
end

function translate_to_sat{T}(initial::T, transition::Dict, goal::T, time::Int64, state_dict::Dict, action_dict::Dict)
    local clauses::Array{Expression, 1} = Array{Expression, 1}();
    local states::AbstractVector = collect(keys(transition));
    local state_count_iterator = countfrom();
    local state_number::Int64 = -1;
    for state in states
        for t in 0:time
            state_number = next(state_count_iterator, state_number)[2];
            state_dict[(state, t)] = Expression(@sprintf("State_%lli", state_number));
        end
    end
    push!(clauses, state_dict[(initial, 0)]);
    push!(clauses, state_dict[(goal, time)]);
    local transition_count_iterator = countfrom();
    local transition_number = -1;
    for state in states
        for action in keys(transition[state])
            state_val_key = transition[state][action];
            for t in 0:(time - 1)
                transition_number = next(transition_count_iterator, transition_number)[2];
                action_dict[(state, action, t)] = Expression(@sprintf("Transition_%lli", transition_number));
                push!(clauses, Expression("==>", action_dict[(state, action, t)], state_dict[(state, t)]));
                push!(clauses, Expression("==>", action_dict[(state, action, t)], state_dict[(state_val_key, (t + 1))]));
            end
        end
    end
    for t in 0:time
        push!(clauses, associate("|", Tuple((collect(state_dict[(state, t)] for state in states)...))));
        for (i, state) in enumerate(states)
            for state_val_key in states[(i+1):end]
                push!(clauses, Expression("|",
                                            Expression("~", state_dict[(state, t)]),
                                            Expression("~", state_dict[(state_val_key, t)])));
            end
        end
    end
    for t in 0:(time - 1)
        local transitions_t::AbstractVector = collect(transition_t for transition_t in keys(action_dict) if (transition_t[3] == t));
        push!(clauses, associate("|", Tuple((collect(action_dict[transition_t] for transition_t in transitions_t)...))));
        for (i, transition_t) in enumerate(transitions_t)
            for transition_t_alternative in transitions_t[(i + 1):end]
                push!(clauses, Expression("|",
                                            Expression("~", action_dict[transition_t]),
                                            Expression("~", action_dict[transition_t_alternative])));
            end
        end
    end
    return associate("&", Tuple((clauses...)));
end

function extract_solution(model::Dict, action_dict::Dict)
    # Collect transitions that are true in the SAT 'model' solution.
    local transitions::AbstractVector = collect(transition for transition in keys(action_dict) if (model[action_dict[transition]]));
    sort!(transitions, lt=(function(t1, t2)
                                return isless(t1[3], t2[3]);
                            end));
    return collect(action for (state, action, time) in transitions);
end

"""
    sat_plan(initial, transition, goal, t_max)

Apply the SATPlan algorithm (Fig. 7.22) to the given planning problem, returning a solution or 'nothing'
when the algorithm fails. The planning problem is converted to conjunctive normal form logic sentence in
order to be solved as a satisfication problem.

The 'initial' and 'goal' states should have the same type.
"""
function sat_plan{T}(initial::T, transition::Dict, goal::T, t_max::Int64; sat_solver::Function=dpll_satisfiable)
    local states::Dict = Dict();
    local actions::Dict = Dict();
    for t in 0:(t_max - 1)
        empty!(states);
        empty!(actions);
        local cnf::Expression = translate_to_sat(initial, transition, goal, t, states, actions);
        local model = sat_solver(cnf);
        if (model != false)
            return extract_solution(model, actions);
        end
    end
    nothing;
end

function is_logic_symbol(s::String)
    if (length(s) == 0)
        return false;
    else
        return isalpha(s[1]);
    end
end

function is_logic_variable_symbol(s::String)
    return (is_logic_symbol(s) && islower(s[1]));
end

function is_logic_variable(e::Expression)
    return ((length(e.arguments) == 0) && islower(e.operator[1]))
end

"""
    is_logic_proposition_symbol(s)

Return if the given 's' is an initial uppercase String that is not 'TRUE' or 'FALSE'.
"""
function is_logic_proposition_symbol(s::String)
    return (is_logic_symbol(s) && isupper(s[1]) && (s != "TRUE") && (s != "FALSE"));
end

#=

    The Python implementation of expr() uses of eval() and overloaded binary/unary infix
    operators to evaluate a String as an Expression.

    This Julia implementation of expr() parses the given String into an expression tree of
    tokens, before returning the parsed Expression.

    Consecutive operators should be delimited with with a space or parentheses.

=#

"""
    expr(s::String)

Parse the given String as an Expression and return the parsed Expression.
"""
function expr(s::String)
    local tokens::AbstractVector = identify_tokens(s);
    tokens = parenthesize_tokens(tokens);
    tokens = parenthesize_arguments(tokens);
    local root_node::ExpressionNode = construct_expression_tree(tokens);
    root_node = prune_nodes(root_node);
    return evaluate_expression_tree(root_node);
end

function expr(e::Expression)
    return e;
end

function subexpressions(e::Expression)
    local answer::AbstractVector = [e];
    for arg in e.arguments
        answer = vcat(answer, subexpressions(arg));
    end
    return answer;
end

function subexpressions(e::Int)
    local answer::AbstractVector = [Expression(string(e))];
    return answer;
end


function variables(e::Expression)
    return Set(x for x in subexpressions(e) if is_logic_variable(x));
end

function proposition_symbols(e::Expression)
    if (is_logic_proposition_symbol(e.operator))
        return [e];
    else
        symbols::Set{Expression} = Set{Expression}();
        for argument in e.arguments
            argument_symbols = proposition_symbols(argument);
            for symbol in argument_symbols
                push!(symbols, symbol);
            end
        end
        return collect(symbols);
    end
end

function is_logic_definite_clause(e::Expression)
    if (is_logic_symbol(e.operator))
        return true;
    elseif (e.operator == "==>")
        antecedent, consequent = e.arguments;
        return (is_logic_symbol(consequent.operator) &&
                all(collect(is_logic_symbol(arg.operator) for arg in conjuncts(antecedent))));
    else
        return false;
    end
end

function parse_logic_definite_clause(e::Expression)
    if (!is_logic_definite_clause(e))
        error("parse_logic_definite_clause: The expression given is not a definite clause!");
    else
        if (is_logic_symbol(e.operator))
            return Array{Expression, 1}([]), e;
        else
            antecedent, consequent = e.arguments;
            return conjuncts(antecedent), consequent;
        end
    end
end

"""
    tt_entails(kb::Expression, alpha::Expression)

Use the truth-table enumeration algorithm (Fig. 7.10) on the sentence 'alpha'
and a conjunction of clauses 'kb'. Return true or false based on whether 'kb' entails
'alpha'.
"""
function tt_entails(kb::Expression, alpha::Expression)
    if (length(variables(alpha)) != 0)
        error("tt_entails(): Found logic variables in 'alpha' Expression!");
    end
    return tt_check_all(kb, alpha, proposition_symbols(Expression("&", kb, alpha)), Dict());
end

function tt_check_all(kb::Expression, alpha::Expression, symbols::AbstractVector, model::Dict)
    if (length(symbols) == 0)
        eval_kb = pl_true(kb, model=model)
        if (typeof(eval_kb) <: Void)
            return true;
        elseif (eval_kb == false)
            return true;
        else
            result = pl_true(alpha, model=model);
            if (typeof(result) <: Bool)
                return result;
            else
                error("tt_check_all(): pl_true() returned an unexpected ", typeof(result), " type!");
            end
        end
    else
        P = symbols[1];
        rest::AbstractVector = symbols[2:end];
        return (tt_check_all(kb, alpha, rest, extend(model, P, true)) &&
                tt_check_all(kb, alpha, rest, extend(model, P, false)));
    end
end

function tt_true(alpha::Expression)
    return tt_entails(Expression("TRUE"), alpha);
end

function tt_true(alpha::String)
    return tt_entails(Expression("TRUE"), expr(alpha));
end

function pl_true(e::Expression; model::Dict=Dict())
    if (e == Expression("TRUE"))
        return true;
    elseif (e == Expression("FALSE"))
        return false;
    elseif (is_logic_proposition_symbol(e.operator))
        return get(model, e, nothing);
    elseif (e.operator == "~")
        subexpression = pl_true(e.arguments[1], model=model);
        if (typeof(subexpression) <: Void)
            return nothing;
        else
            return !subexpression;
        end
    elseif (e.operator == "|")
        result = false;
        for argument in e.arguments
            subexpression = pl_true(argument, model=model);
            if (subexpression == true)
                return true;
            end
            if (typeof(subexpression) <: Void)
                result = nothing;
            end
        end
        return result;
    elseif (e.operator == "&")
        result = true;
        for argument in e.arguments
            subexpression = pl_true(argument, model=model);
            if (subexpression == false)
                return false;
            end
            if (typeof(subexpression) <: Void)
                result = nothing;
            end
        end
        return result;
    end
    local p::Expression;
    local q::Expression;
    if (length(e.arguments) == 2)
        p, q = e.arguments;
    else
        error("PropositionalLogicError: Expected 2 arguments in expression ", repr(e),
                " got ", length(e.arguments), " arguments!");
    end
    if (e.operator == "==>")
        return pl_true(Expression("|", Expression("~", p), q), model=model);
    elseif (e.operator == "<==")
        return pl_true(Expression("|", p, Expression("~", q)), model=model);
    end;
    p_t = pl_true(p, model=model);
    if (typeof(p_t) <: Void)
        return nothing;
    end
    q_t = pl_true(q, model=model);
    if (typeof(q_t) <: Void)
        return nothing;
    end
    if (e.operator == "<=>")
        return p_t == q_t;
    elseif (e.operator == "^")
        return p_t != q_t;
    else
        error("PropositionalLogicError: Illegal operator detected in expression ", repr(e), "!")
    end
end

"""
    eliminate_implications(e)

Eliminate any implications in the given Expression by using the definition of biconditional introduction,
material implication, and converse implication with De Morgan's Laws and return the modified Expression.
"""
function eliminate_implications(e::Expression)
    if ((length(e.arguments) == 0) || is_logic_symbol(e.operator))
        return e;
    end
    local arguments = map(eliminate_implications, e.arguments);
    local a::Expression = first(arguments);
    local b::Expression = last(arguments);
    if (e.operator == "==>")
        return Expression("|", b, Expression("~", a));
    elseif (e.operator == "<==")
        return Expression("|", a, Expression("~", b));
    elseif (e.operator == "<=>")
        return Expression("&", Expression("|", a, Expression("~", b)), Expression("|", b, Expression("~", a)));
    elseif (e.operator == "^")
        if (length(arguments) != 2)
            #If the length of 'arguments' is 1, last(arguments)
            #gives us the same Expression for 'b' as 'a'.
            error("EliminateImplicationsError: XOR should be applied to 2 arguments, found ",
                length(arguments), " arguments!");
        end
        return Expression("|", Expression("&", a, Expression("~", b)), Expression("&", Expression("~", a), b));
    else
        if (!(e.operator in ("&", "|", "~")))
            error("EliminateImplicationsError: Found an unexpected operator '", e.operator, "'!");
        end
        return Expression(e.operator, arguments...);
    end
end

function move_not_inwards_negate_argument(e::Expression)
    return move_not_inwards(Expression("~", e));
end

"""
    move_not_inwards(e)

Apply De Morgan's laws to the given Expression and return the modified Expression.
"""
function move_not_inwards(e::Expression)
    if (e.operator == "~")
        local a::Expression = e.arguments[1];
        if (a.operator == "~")
            return move_not_inwards(a.arguments[1]);
        elseif (a.operator == "&")
            return associate("|", map(move_not_inwards_negate_argument, a.arguments));
        elseif (a.operator == "|")
            return associate("&", map(move_not_inwards_negate_argument, a.arguments));
        else
            return e;
        end
    elseif (is_logic_symbol(e.operator) || (length(e.arguments) == 0))
        return e;
    else
        return Expression(e.operator, map(move_not_inwards, e.arguments)...);
    end
end

function distribute_and_over_or(e::Expression)
    if (e.operator == "|")
        local a::Expression = associate("|", e.arguments);
        if (a.operator != "|")
            return distribute_and_over_or(a);
        elseif (length(a.arguments) == 0)
            return Expression("FALSE");
        elseif (length(a.arguments) == 1)
            return distribute_and_over_or(a.arguments[1]);
        end
        conjunction = findfirst((function(arg)
                            return (arg.operator == "&");
                        end), a.arguments);
        if (conjunction == 0)  #(&) operator was not found in a.arguments
            return a;
        else
            conjunction = a.arguments[conjunction];
        end
        others = Tuple((collect(a for a in a.arguments if (!(a == conjunction)))...));
        rest = associate("|", others);
        return associate("&", Tuple((collect(distribute_and_over_or(Expression("|", conjunction_arg, rest))
                                    for conjunction_arg in conjunction.arguments)...)));
    elseif (e.operator == "&")
        return associate("&", map(distribute_and_over_or, e.arguments));
    else
        return e;
    end
end

function expand_prefix_nary_expression(operator::String, arguments::AbstractVector)
    if (length(arguments) == 1)
        return arguments[1];
    else
        current = first(arguments);
        rest = arguments[2:end];
        return Expression(operator, current, expand_prefix_nary_expression(operator, rest));
    end
end

function associate(operator::String, arguments::Tuple)
    dissociated_arguments = dissociate(operator, arguments);
    if (length(dissociated_arguments) == 0)
        if (operator == "&")
            return Expression("TRUE");
        elseif (operator == "|")
            return Expression("FALSE");
        elseif (operator == "+")
            return Expression("0");
        elseif (operator == "*")
            return Expression("1");
        else
            error("AssociateError: Found unexpected operator '", operator, "'!");
        end
    elseif (length(dissociated_arguments) == 1)
        return dissociated_arguments[1];
    else
        return Expression(operator, Tuple((dissociated_arguments...)));
    end
end

function dissociate_collect(operator::String, arguments::Tuple{Vararg{Expression}}, result_array::AbstractVector)
    for argument in arguments
        if (argument.operator == operator)
            dissociate_collect(operator, argument.arguments, result_array);
        else
            push!(result_array, argument);
        end
    end
    nothing;
end

function dissociate(operator::String, arguments::Tuple{Vararg{Expression}})
    local result = Array{Expression, 1}([]);
    dissociate_collect(operator, arguments, result);
    return result;
end

function conjuncts(e::Expression)
    return dissociate("&", (e,));
end

function disjuncts(e::Expression)
    return dissociate("|", (e,));
end

"""
    to_conjunctive_normal_form(s)

Convert the given propositional logical sentence 's' to its equivalent
conjunctive normal form.
"""
function to_conjunctive_normal_form(sentence::Expression)
    return distribute_and_over_or(move_not_inwards(eliminate_implications(sentence)));
end

function to_conjunctive_normal_form(sentence::String)
    return distribute_and_over_or(move_not_inwards(eliminate_implications(expr(sentence))));
end

function pl_resolve(c_i::Expression, c_j::Expression)
    local clauses = Array{Expression, 1}([]);
    for d_i in disjuncts(c_i)
        for d_j in disjuncts(c_j)
            if ((d_i == Expression("~", d_j)) || (Expression("~", d_i) == d_j))
                d_new = Tuple((collect(Set{Expression}(append!(removeall(disjuncts(c_i), d_i), removeall(disjuncts(c_j), d_j))))...));
                push!(clauses, associate("|", d_new));
            end
        end
    end
    return clauses;
end

"""
    pl_resolution(kb, alpha)

Apply a simple propositional logic resolution algorithm (Fig. 7.12) on the given knowledge base
and propositional logic sentence (query). Return a boolean indicating if the sentence follows
the clauses that exist in the given knowledge base.
"""
function pl_resolution{T <: AbstractKnowledgeBase}(kb::T, alpha::Expression)
    local clauses::AbstractVector = append!(copy(kb.clauses), conjuncts(to_conjunctive_normal_form(Expression("~", alpha))));
    local new_set = Set{Expression}();
    while (true)
        n = length(clauses);
        pairs = collect((clauses[i], clauses[j]) for i in 1:n for j in i+1:n);
        for (c_i, c_j) in pairs
            local resolvents = pl_resolve(c_i, c_j);
            if (Expression("FALSE") in resolvents)
                return true;
            end
            union!(new_set, Set{Expression}(resolvents));
        end
        if (issubset(new_set, Set{Expression}(clauses)))
            return false;
        end
        for c in new_set
            if (!(c in clauses))
                push!(clauses, c);
            end
        end
    end
end

function occurrence_check(key, x, substitutions::Dict)
    if (key == x)
        println("occurrence_check(::Any, ::Any, ::Dict) returned true!!!");
        return true;
    else
        return false;
    end
end

function occurrence_check{T <: Union{Tuple, AbstractVector}}(key::Expression, x::T, substitutions::Dict)
    if (key == x)
        println("occurrence_check(::Expression, ::Union{Tuple, AbstractVector}, ::Dict) returned true!!!");
        return true;
    else
        if (length(collect(true for element in x if (occurrence_check(key, element, substitutions)))) == 0)
            return false;
        else
            return true;
        end
    end
end

function occurrence_check(key::Expression, x::Expression, substitutions::Dict)
    if (key == x)
        println("occurrence_check(::Expression, ::Expression, ::Dict) returned true!!!");
        return true;
    elseif (is_logic_variable(x) && (x in keys(substitutions)))
        return occurrence_check(key, substitutions[x], substitutions);
    else
        return (occurrence_check(key, x.operator, substitutions) ||
                occurrence_check(key, x.arguments, substitutions));
    end
end

function unify_variable(key::Expression, x::Expression, substitutions::Dict)
    if (key in keys(substitutions))
        return unify(substitutions[key], x, substitutions);
    elseif (x in keys(substitutions))
        return unify(key, substitutions[x], substitutions);
    elseif (occurrence_check(key, x, substitutions))
        return nothing;
    else
        return extend(substitutions, key, x);
    end
end

"""
    unify(e1, e2, substitutions)

Return a dictionary of substitutions that make 'e1' equal to 'e2' using the unification algorithm (Fig. 9.1)
and return 'nothing' on failure.
"""
unify(e1, e2, substitutions::Void) = nothing;
unify(e1::Expression, e2::String, substitutions::Dict) = nothing;

function unify(e1::String, e2::String, substitutions::Dict)
    if (e1 == e2)
        return substitutions;
    else
        return nothing;
    end
end

function unify(e1::Expression, e2::Expression, substitutions::Dict)
    if (e1 == e2)
        return substitutions;
    elseif (is_logic_variable(e1))
        return unify_variable(e1, e2, substitutions);
    elseif (is_logic_variable(e2))
        return unify_variable(e2, e1, substitutions);
    else
        return unify(e1.arguments, e2.arguments, unify(e1.operator, e2.operator, substitutions));
    end
end

function unify{S <: Union{AbstractVector, Tuple}, T <: Union{AbstractVector, Tuple}}(a1::S, a2::T, substitutions::Dict)
    if (a1 == a2)
        return substitutions;
    else
        if (length(a1) == length(a2))
            return unify(a1[2:end], a2[2:end], unify(a1[1], a2[1], substitutions));
        else
            return nothing;
        end
    end
end

"""
    extend(dict, key, val)

Make a copy of the given dict and overwrite any existing value corresponding
to the 'key' as 'val' and return the new dictionary.
"""
function extend(dict::Dict, key, val)
    local new_dict::Dict = copy(dict);
    new_dict[key] = val;
    return new_dict;
end

function substitute(dict::Dict, e)
    if (typeof(e) <: AbstractVector)
        return collect(substitute(dict, element) for element in e);
    elseif (typeof(e) <: Tuple)
        return (collect(substitute(dict, element) for element in e)...);
    else
        return e;
    end
end

function substitute(dict::Dict, e::Expression)
    if (is_logic_variable_symbol(e.operator))
        return get(dict, e, e);
    else
        return Expression(e.operator, collect(substitute(dict, argument) for argument in e.arguments)...);
    end
end

# This number is used to generate new variables in standardize_variables().
standardize_variables_counter = [countfrom(BigInt(1)), BigInt(-1)];

function standardize_variables(e, counter::AbstractVector; dict::Union{Void, Dict}=nothing)
    return e;
end

function standardize_variables(e::Expression, counter::AbstractVector; dict::Union{Void, Dict}=nothing)
    if (typeof(dict) <: Void)
        dict = Dict();
    end
    if (is_logic_variable_symbol(e.operator))
        if (haskey(dict, e))
            return dict[e]
        else
            counter[2] = next(counter[1], counter[2])[2];
            local val::Expression = Expression(@sprintf("v_%s", repr(counter[2])));
            dict[e] = val;
            return val;
        end
    else
        return Expression(e.operator, collect(standardize_variables(argument, counter, dict=dict) for argument in e.arguments)...);
    end
end

"""
    fol_fc_ask(kb, alpha)

Apply a simple forward-chaining algorithm (Fig. 9.3) to a atomic sentence query and knowledge base of
first-order logic definite clauses. Return a dictionary of substitutions or false on failure.
"""
function fol_fc_ask(kb::FirstOrderLogicKnowledgeBase, alpha::Expression)
    local new_sentences::Array{Expression, 1} = Array{Expression, 1}();
    while (length(new_sentences) == 0)
        empty!(new_sentences);
        for rule in kb.clauses
            p, q = parse_logic_definite_clause(standardize_variables(rule, standardize_variables_counter));
            # Generate p_prime using unimplemented invert_literals() function.
            for p_prime in fetch(invert_literals(kb.clauses))
                # generate_thetas() is an unimplmented function that generates substitutions
                # such that each dictionary of substitution 's' fulfills 
                # (substitute(s, p) == substitute(s, p_prime))
                for theta in generate_thetas(p, p_prime)
                    q_prime = substitute(theta, q);
                    # Check if either q' and 'atomic_sentences(kb)''
                    # or q' and 'new_sentences' can unify.
                    #
                    # Both unify(::Expression, ::Expression) and atomic_sentences(kb)
                    # are unimplemented.
                    if (!(typeof(unify(q_prime, atomic_sentences(kb))) <: Void) ||
                        !(typeof(unify(q_prime, new_sentences)) <: Void))
                        push!(new_sentence, q_prime);
                        phi = unify(q_prime, alpha);
                        if (!(typeof(phi) <: Void))
                            return phi;
                        end
                    end
                end
            end
        end
        # Add atomic sentences in new_sentences to 'kb'.
        for new_sentence in new_sentences
            for new_literal in atomic_sentences(new_sentence)
                tell(kb, new_literal);
            end
        end
    end
    return false;
end

function fol_bc_and(kb::FirstOrderLogicKnowledgeBase, goals::Array{Expression, 1}, theta::Void, standardize_variables_counter_ref::AbstractVector)
    return ();
end

function fol_bc_and(kb::FirstOrderLogicKnowledgeBase, goals::Array{Expression, 1}, theta::Dict, standardize_variables_counter_ref::AbstractVector)
    if (length(goals) == 0)
        return (theta,);
    else
        first, rest = goals[1], goals[2:end];
        thetas = ();
        for theta_prime in fol_bc_or(kb, substitute(theta, first), theta, standardize_variables_counter_ref)
            for theta_prime_prime in fol_bc_and(kb, rest, theta_prime, standardize_variables_counter_ref)
                thetas = Tuple((thetas..., theta_prime_prime));
            end
        end
        return thetas;
    end

end

function fol_bc_or(kb::FirstOrderLogicKnowledgeBase, goal::Expression, theta::Dict, standardize_variables_counter_ref::AbstractVector)
    thetas = ();
    for rule in fetch_rules_for_goal(kb, goal)
        lhs, rhs = parse_logic_definite_clause(standardize_variables(rule, standardize_variables_counter_ref));
        for theta_prime in fol_bc_and(kb, lhs, unify(rhs, goal, theta), standardize_variables_counter_ref)
            thetas = Tuple((thetas..., theta_prime));
        end
    end
    return thetas;
end

"""
    fol_bc_ask(kb, query)

Use a simple backward-chaining algorithm (Fig. 9.6) for first-order knowledge bases 
on the given knowledge base and atomic sentence 'query' to return an array of
dictionaries of substitutions.
"""
function fol_bc_ask(kb::FirstOrderLogicKnowledgeBase, query::Expression)
    return fol_bc_or(kb, query, Dict(), standardize_variables_counter);
end

function is_number(n::String)   # Not to be confused with isnumber() from Julia's Base library.
    if (length(n) == 0)
        return false;
    else
        if (isnumber(strip(n)))
            return true;
        else
            local period_count = count((function(c::Char)
                                            return c == '.';
                                        end), n);
            if (period_count == 1)
                local decimal_index::UInt64 = UInt64(search(n, '.'));
                if (decimal_index == UInt64(0))
                    error("is_number(): Could not find decimal point!");
                else
                    return (isnumber(n[1:(decimal_index - UInt64(1))]) && isnumber(n[(decimal_index + UInt64(1)):end]))
                end
            else
                return false
            end
        end
    end
end

#=

    The functions differentiate(), simplify(), and differentiate_simplify() are not in the book.

    These functions apply symbolic differentiation on Expressions.

    Symbolic differentiation used to be part of AI.

    Now this topic is considered a separate field, Symbolic Algebra.

=#
function differentiate(y::Expression, x::Expression)
    if (x == y)
        return Expression("1");
    elseif (length(y.arguments) == 0)
        return Expression("0");
    else
        local u::Expression = y.arguments[1];
        local operator::String = y.operator;
        local v::Expression = reverse(y.arguments)[1];
        if (operator == "+")
            return Expression("+", differentiate(u, x), differentiate(v, x));
        elseif ((operator == "-") && (length(y.arguments) == 1))
            return Expression("-", differentiate(u, x));
        elseif (operator == "-")
            return ExpressioN("-", differentiate(u, x), differentiate(v, x));
        elseif (operator == "*")
            return Expression("+",
                            Expression("*", u, differentiate(v, x)),
                            Expression("*", v, differentiate(u, x)));
        elseif (operator == "/")
            return Expression("/",
                            Expression("-",
                                        Expression("*", v, differentiate(u, x)),
                                        Expression("*", u, differentiate(v, x))),
                            Expression("*", v, v));
        elseif ((operator == "^") && is_number(x.operator))
            return Expression("*",
                                Expression("^",
                                            Expression("*", v, u),
                                            Expression("-", v, Expression("1"))),
                                differentiate(u, x));
        elseif (operator == "^")
            return Expression("+",
                                Expression("*",
                                            Expression("^",
                                                        Expression("*", v, u),
                                                        Expression("-", v, Expression("1"))),
                                            differentiate(u, x)),
                                Expression("*",
                                            Expression("*",
                                                        Expression("^", u, v),
                                                        Expression("log", u)),
                                            differentiate(v, x)));
        elseif (operator == "log")
            return Expression("/", differentiate(u, x), u);
        else
            error("differentiate(): Unknown operator ", operator, " in differentiate(", repr(y), repr(x), ")!");
        end
    end
end

function simplify(x::Expression)
    if (is_number(x.operator))
        return x
    elseif (length(x.arguments) == 0)
        return x;
    end
    local arguments::Tuple = map(simplify, x.arguments);
    local u::Expression = arguments[1];
    local operator::String = x.operator;
    local v::Expression = reverse(arguments)[1];
    if (operator == "+")
        if (v == Expression("0"))
            return u;
        elseif (u == Expression("0"))
            return v;
        elseif (u == v)
            return Expression("*", Expression("2"), u);
        elseif ((u == Expression("-", v)) || (v == Expression("-", u)))
            return Expression("0");
        end
    elseif ((operator == "-") && (length(arguments) == 1))
        if ((u.operator == "-") && (length(arguments) == 1))
            return u.arguments[1];
        end
    elseif (operator == "-")
        if (v == Expression("0"))
            return u;
        elseif (u == Expression("0"))
            return Expression("-", v);
        elseif (u == v)
            return Expression("0");
        elseif ((u == Expression("-", v)) || (v == Expression("-", u)))
            return Expression("0");
        end
    elseif (operator == "*")
        if ((u == Expression("0")) || (v == Expression("0")))
            return Expression("0");
        elseif (u == Expression("1"))
            return v;
        elseif (v == Expression("1"))
            return u;
        elseif (u == v)
            return Expression("^", u, Expression("2"));
        end
    elseif (operator == "/")
        if (u == Expression("0"))
            return Expression("0");
        elseif (v == Expression("0"))
            return Expression("Inf");
        elseif (u == v)
            return Expression("1");
        elseif ((u == Expression("-", v)) || (v == Expression("-", u)))
            return Expression("0");
        end
    elseif (operator == "^")
        if (u == Expression("0"))
            return Expression("0");
        elseif (v == Expression("0"))
            return Expression("1");
        elseif (u == Expression("1"))
            return Expression("1");
        elseif (v == Expression("1"))
            return v;
        end
    elseif (operator == "log")
        if (u == Expression("1"))
            return Expression("0");
        end
    else
        error("simplify(): Found unknown operator ", operator, "!");
    end
    return Expression(operator, arguments...);
end

function differentiate_simplify(y::Expression, x::Expression)
    return simplify(differentiate(y, x));
end

type ExpressionNode
    value::Nullable{String}
    parent::Nullable{ExpressionNode}
    children::Array{ExpressionNode, 1}

    function ExpressionNode(;val::Union{Void, String}=nothing, parent::Union{Void, ExpressionNode}=nothing)
        return new(Nullable{String}(val), Nullable{ExpressionNode}(parent), []);
    end
end

function identify_tokens(s::String)
    local existing_parenthesis::Int64 = 0;
    local queue::Array{String, 1} = Array{String, 1}([]);
    local current_string::Array{Char, 1} = Array{Char, 1}([]);
    local isOperator::Bool = false;
    for character in s
        if (character == '(')
            existing_parenthesis = existing_parenthesis + 1;

            if (strip(String(current_string)) != "")
                push!(queue, strip(String(current_string)));
            end
            push!(queue, "(");

            if (isOperator)
                isOperator = false;
            end
            current_string = Array{Char, 1}([]);
        elseif (character == ')')
            existing_parenthesis = existing_parenthesis - 1;

            if (strip(String(current_string)) != "")
                push!(queue, strip(String(current_string)));
            end
            push!(queue, ")");

            if (isOperator) #operators can't be leaves
                error("ConstructExpressionTreeError: Detected operator at leaf level!");
            end

            current_string = Array{Char, 1}([]);
        elseif (character == ',')
            if (strip(String(current_string)) == "")
                if (queue[length(queue)] == ")")    #do nothing
                else
                    error("ConstructExpressionTreeError: Invalid n-Tuple detected!");
                end
            else
                push!(queue, strip(String(current_string)));
            end

            push!(queue, ",");

            current_string = Array{Char, 1}([]);
        elseif (character == ' ')   #white space is considered
            if (isOperator)
                push!(queue, strip(String(current_string)));
                current_string = Array{Char, 1}([]);
                isOperator = false;
            end

            push!(current_string, character);
        elseif (character in ('+', '-', '*', '/', '\\', '=', '<', '>', '\$', '|', '%', '^', '~', '&', '?'))
            if (!isOperator)
                if (strip(String(current_string)) != "")
                    push!(queue, strip(String(current_string)));
                end
                current_string = Array{Char, 1}([]);
            end
            push!(current_string, character);
            isOperator = true;
        else    #found new symbol  
            if (isOperator) #first character of new token
                push!(queue, strip(String(current_string)));
                current_string = Array{Char, 1}([]);
                isOperator = false;
            end
            push!(current_string, character);
        end


        if (existing_parenthesis < 0)
            error("ConstructExpressionTreeError: Invalid parentheses syntax detected!");
        end
    end
    #Check for a possible token at the end of the String.
    if (strip(String(current_string)) != "")
        push!(queue, strip(String(current_string)));
    end

    if (existing_parenthesis != 0)
        error("ConstructExpressionTreeError: Invalid number of parentheses!");
    end
    return queue;
end

#Parenthesize any arguments that are not enclosed by parentheses
function parenthesize_arguments(tokens::AbstractVector) 
    local existing_parenthesis::Int64 = 0;
    local comma_indices::Array{Int64, 1} = Array{Int64, 1}([]);
    #keep track of opening and closing parentheses indices
    #keep track of comma indices at the same tree level
    #this function runs after parenthesize_tokens()
    for index in 1:length(tokens)
        if (tokens[index] == ",")
            push!(comma_indices, index);
        end
    end
    for comma_index in reverse(comma_indices)
        ####println("index to modify: ", comma_index, " token: ", tokens[comma_index],
        ####        " comma indices: ", comma_indices, " tokens: ", tokens...);
        no_parentheses = false; #boolean indicating if the current argument is enclosed in parentheses
        #,=>    #the order of indices to search rightward
        existing_parenthesis = 0;
        for index in (comma_index + 1):length(tokens)
            if (tokens[index] == "(")
                existing_parenthesis = existing_parenthesis + 1;
            elseif (tokens[index] == ")")
                existing_parenthesis = existing_parenthesis - 1;
                if (index == (comma_index + 1))
                    error("ConstructExpressionTreeError: Found ')', expected argument!");
                end
            elseif (tokens[index] == ",")
                if (existing_parenthesis == 0)  #found following comma in same tree level
                    #Add parentheses
                    if (no_parentheses)
                        insert!(tokens, index, ")");
                        insert!(tokens, (comma_index + 1), "(");
                    end
                    break;
                end
            else
                if (existing_parenthesis == 0)  #the current argument is not enclosed in parentheses
                    no_parentheses = true;
                end
            end

            if (existing_parenthesis == -1) #found end of arguments for infix function
                ####println("index: ", index, " token: ", tokens[index], " no_parentheses: ", no_parentheses);
                #Add parentheses
                if (no_parentheses)
                    insert!(tokens, index, ")");
                    insert!(tokens, (comma_index + 1), "(");
                end
                break;
            end
        end
        no_parentheses = false; #boolean indicating if the current argument is enclosed in parentheses
        #<=,    #reverse the order of indices to search leftward
        existing_parenthesis = 0;
        for index in reverse(1:(comma_index - 1))
            if (tokens[index] == "(")
                existing_parenthesis = existing_parenthesis + 1;
                if (index == (comma_index - 1))
                    error("ConstructExpressionTreeError: Found '(', expected argument!");
                end
            elseif (tokens[index] == ")")
                existing_parenthesis = existing_parenthesis - 1;
            elseif (tokens[index] == ",")
                if (existing_parenthesis == 0)  #found following comma in same tree level
                    #Add parentheses
                    if (no_parentheses)
                        insert!(tokens, comma_index, ")");
                        insert!(tokens, (index + 1), "(");
                    end
                    break;
                end
            else
                if (existing_parenthesis == 0)  #the current argument is not enclosed in parentheses
                    no_parentheses = true;
                end
            end

            if (existing_parenthesis == 1) #found end of arguments for infix function
                ####println("index: ", index, " token: ", tokens[index], " no_parentheses: ", no_parentheses);
                #Add parentheses
                if (no_parentheses)
                    insert!(tokens, comma_index, ")");
                    insert!(tokens, index, "(");
                end
                break;
            end
        end
    end
    return tokens;
end

function parenthesize_tokens(tokens::AbstractVector)
    local existing_parenthesis::Int64 = 0;
    local add_parentheses_at::Array{Int64, 1} = Array{Int64, 1}([]);   #-1 if nothing should be done
    #Find next prefix operator without a following '('
    for index in 1:length(tokens)
        if (any((function(c::Char)
                        return c in tokens[index];
                    end),
                    ('+', '-', '*', '/', '\\', '=', '<', '>', '\$', '|', '%', '^', '~', '&', '?')))
            #Check if '(' exists already
            if (((index + 1) != length(tokens) + 1) && (tokens[index + 1] != "("))
                push!(add_parentheses_at, index);
            end
        end
    end
    for last_entry_index in reverse(add_parentheses_at)
        ####println("index to modify: ", last_entry_index, " token: ", tokens[last_entry_index],
        ####        " tokens: ", tokens...);
        modified_tokens::Bool = false;
        for index in (last_entry_index + 1):length(tokens)
            if (tokens[index] == "(")
                existing_parenthesis = existing_parenthesis + 1;
            elseif (tokens[index] == ")")
                existing_parenthesis = existing_parenthesis - 1;
            end
            if (existing_parenthesis == 0)
                if (((index + 1) < length(tokens)) &&   #'(' should not exist at the end of the expression
                    (tokens[index + 1] != "("))
                    insert!(tokens, index + 1, ")");
                    insert!(tokens, last_entry_index + 1, "(");
                    modified_tokens = true;
                    break;
                elseif (index == length(tokens))
                    insert!(tokens, index + 1, ")");
                    insert!(tokens, last_entry_index + 1, "(");
                    modified_tokens = true;
                    break;
                end
            elseif (existing_parenthesis == -1) #reached higher tree level (')'), ('(') should exist
                insert!(tokens, index, ")");
                insert!(tokens, last_entry_index + 1, "(");
                existing_parenthesis = 0;
                modified_tokens = true;
                break;
            end
        end
        if (!modified_tokens)
            error("ConstructExpressionTreeError: Could not add parentheses to the expression!");
        end
    end
    return tokens;
end

function construct_expression_tree(tokens::AbstractVector)
    local existing_parenthesis::Int64 = 0;
    local current_node::ExpressionNode = ExpressionNode();
    local root_node::ExpressionNode = current_node;
    local unary_depth::Int64 = 0;   #when operator exists and we traverse to a new child node
    for token in tokens
        if (token == "(")
            existing_parenthesis = existing_parenthesis + 1;

            #Create new level and visit it
            new_node = ExpressionNode(parent=current_node);
            push!(current_node.children, new_node);
            current_node = new_node;
        elseif (token == ")")
            existing_parenthesis = existing_parenthesis - 1;
            if (!isnull(current_node.parent))
                current_node = get(current_node.parent);
            else
                error("ConstructExpressionTreeError: The root node does not have a parent!");
            end
        elseif (token == ",")
            if (!isnull(current_node.value) && get(current_node.value) != ",")
                notFound = true;
                
                new_intermediate_node = ExpressionNode(val=token, parent=get(current_node.parent));
                for (i, c) in enumerate(get(current_node.parent).children)
                    if (c == current_node)
                        deleteat!(get(current_node.parent).children, i);
                        insert!(get(current_node.parent).children, i, new_intermediate_node);
                        notFound = false;
                        break;
                    end
                end
                if (notFound)
                    error("ConstructExpressionTreeError: could not find existing child node!");
                end
                

                current_node.parent = Nullable{ExpressionNode}(new_intermediate_node);
                push!(new_intermediate_node.children, current_node);
                current_node = new_intermediate_node;
            else
                current_node.value = Nullable{String}(",");
            end
        elseif (any((function(c::Char)
                        return c in token;
                    end),
                    ('+', '-', '*', '/', '\\', '=', '<', '>', '\$', '|', '%', '^', '~', '&', '?')))
            #Check if operator exists already
            if (isnull(current_node.value))
                current_node.value = Nullable{String}(token);
            else
                if (!any((function(c::Char)
                        return c in token;
                    end),
                    ('+', '-', '*', '/', '\\', '=', '<', '>', '\$', '|', '%', '^', '~', '&', '?')))
                    if (isnull(current_node.parent))
                        new_root_node = ExpressionNode(val=token);
                        push!(new_root_node.children, current_node);
                        current_node.parent = new_root_node;
                        current_node = new_root_node;
                    else
                        notFound = true;
                        new_intermediate_node = ExpressionNode(val=token, parent=get(current_node.parent));

                        for (i, c) in enumerate(get(current_node.parent).children)
                            if (c == current_node)
                                deleteat!(get(current_node.parent).children, i);
                                insert!(get(current_node.parent).children, i, new_intermediate_node);
                                notFound = false;
                                break;
                            end
                        end
                        if (notFound)
                            error("ConstructExpressionTreeError: Could not find existing child node!");
                        end

                        current_node.parent = Nullable{ExpressionNode}(new_intermediate_node);
                        push!(new_intermediate_node.children, current_node);
                        current_node = new_intermediate_node;
                    end
                else
                    if (isnull(current_node.parent))
                        new_root_node = ExpressionNode(val=token);
                        current_node.parent = new_root_node;
                        push!(new_root_node.children, current_node);
                        current_node = new_root_node;
                    else
                        notFound = true;
                        new_intermediate_node = ExpressionNode(val=token, parent=get(current_node.parent));

                        for (i, c) in enumerate(get(current_node.parent).children)
                            if (c == current_node)
                                deleteat!(get(current_node.parent).children, i);
                                insert!(get(current_node.parent).children, i, new_intermediate_node);
                                notFound = false;
                                break;
                            end
                        end
                        if (notFound)
                            error("ConstructExpressionTreeError: Could not find existing child node!");
                        end

                        current_node.parent = Nullable{ExpressionNode}(new_intermediate_node);
                        push!(new_intermediate_node.children, current_node);
                        current_node = new_intermediate_node;
                    end
                end
            end
        else    #Not a special operator
            if (isnull(current_node.value))
                current_node.value = Nullable{String}(token);
            else
                new_node = ExpressionNode(val=token, parent=current_node);
                push!(current_node.children, new_node);
            end
        end


        if (existing_parenthesis < 0)
            error("ConstructExpressionTreeError: Invalid parentheses syntax detected!");
        end
    end

    while (!isnull(root_node.parent))
        root_node = get(root_node.parent);
    end

    if (existing_parenthesis != 0)
        error("ConstructExpressionTreeError: Invalid number of parentheses!");
    end
    return root_node;
end

function prune_nodes(node::ExpressionNode)
    #remove valueless nodes that have 1 child
    for child in node.children
        prune_nodes(child);
    end
    if (isnull(node.value))
        if (length(node.children) == 1)
            if (isnull(node.parent))
                new_root_node = pop!(node.children);
                new_root_node.parent = Nullable{ExpressionNode}();
                return new_root_node;
            else
                notFound = true;
                new_node = pop!(node.children);

                for (i, c) in enumerate(get(node.parent).children)
                    if (c == node)
                        deleteat!(get(current_node.parent).children, i);
                        insert!(get(current_node.parent).children, i, new_node);
                        notFound = false;
                        break;
                    end
                end
                if (notFound)
                    error("ConstructExpressionTreeError: Could not find existing child node!");
                end

                new_node.parent = Nullable{ExpressionNode}(current_node.parent);
                return new_node;
            end
        else
            error("ConstructExpressionTreeError: Found ", length(node.children), " children in valueless ExpressionNode!");
        end
    end
    return node;
end

function evaluate_expression_tree(node::ExpressionNode)
    local queue::AbstractVector = [];
    for child in node.children
        if (get(child.value) != ",")
            push!(queue, evaluate_expression_tree(child));
        else #Use current operator for childrens' children
            for child_child in child.children
                push!(queue, evaluate_expression_tree(child_child));
            end
        end
    end
    if (length(node.children) == 0)
        return Expression(get(node.value));
    else
        return Expression(get(node.value), queue...);
    end
end

# A simple inference in a wumpus world (Fig. 7.13)

wumpus_world_inference = expr("(B11 <=> (P12 | P21))  &  ~B11");

# A PropositionalDefiniteKnowledgeBase representation of (Fig 7.16) to be used with
# the propositional logic forward-chaining algorithm pl_fc_entails() (Fig 7.15).

horn_clauses_kb = PropositionalDefiniteKnowledgeBase();

for clause in map(expr, map(String, split("P==>Q; (L&M)==>P; (B&L)==>M; (A&P)==>L; (A&B)==>L; A;B", ";")))
    tell(horn_clauses_kb, clause);
end

test_fol_kb = FirstOrderLogicKnowledgeBase(map(expr, ["Farmer(Mac)",
                                                        "Rabbit(Pete)",
                                                        "Mother(MrsMac, Mac)",
                                                        "Mother(MrsRabbit, Pete)",
                                                        "(Rabbit(r) & Farmer(f)) ==> Hates(f, r)",
                                                        "(Mother(m, c)) ==> Loves(m, c)",
                                                        "(Mother(m, r) & Rabbit(r)) ==> Rabbit(m)",
                                                        "(Farmer(f)) ==> Human(f)",
                                                        # "(Human(h) & Mother(m, h)) ==> Human(m)" # results in infinite recursion
                                                        "(Mother(m, h) & Human(h)) ==> Human(m)"]));

crime_kb = FirstOrderLogicKnowledgeBase(map(expr, ["(American(x) & Weapon(y) & Sells(x, y, z) & Hostile(z)) ==> Criminal(x)",
                                                    "Owns(Nono, M1)",
                                                    "Missile(M1)",
                                                    "(Missile(x) & Owns(Nono, x)) ==> Sells(West, x, Nono)",
                                                    "Missile(x) ==> Weapon(x)",
                                                    "Enemy(x, America) ==> Hostile(x)",
                                                    "American(West)",
                                                    "Enemy(Nono, America)"]));

function pretty_set(s::Set{Expression})
    return @sprintf("Set(%s)", repr(sort(collect(s),
                                        lt=(function(e1::Expression, e2::Expression)
                                                return isless(e1.operator, e2.operator);
                                            end))));
end

