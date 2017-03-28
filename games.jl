
import Base.display;

export AbstractGame, Figure52Game,
        minimax_decision, alphabeta_full_search, alphabeta_search;

abstract AbstractGame;

#=

    Game is an abstract game that contains an initial state.

    Games have a corresponding utility function, terminal test, set of legal moves, and transition model.

=#

type Game <: AbstractGame
    initial::String

    function Game(initial_state::String)
        return new(initial_state);
    end
end

function actions{T <: AbstractGame}(game::T, state::String)
    println("actions() is not implemented yet for ", typeof(game), "!");
    nothing;
end

function result{T <: AbstractGame}(game::T, state::String, move::String)
    println("result() is not implemented yet for ", typeof(game), "!");
    nothing;
end

function utility{T <: AbstractGame}(game::T, state::String, player::String)
    println("utility() is not implemented yet for ", typeof(game), "!");
    nothing;
end

function terminal_test{T <: AbstractGame}(game::T, state::String)
    if (length(actions(game, state)) == 0)
        return true;
    else
        return false;
    end
end

function to_move{T <: AbstractGame}(game::T, state::String)
    println("to_move() is not implemented yet for ", typeof(game), "!");
    nothing;
end

function display{T <: AbstractGame}(game::T, state::String)
    println(state);
end

#=

    Figure52Game is the game represented by the game tree in Fig. 5.2.

=#
type Figure52Game <: AbstractGame
    initial::String
    nodes::Dict
    utilities::Dict

    function Figure52Game()
        return new("A", Dict([
                            Pair("A", Dict("A1"=>"B",  "A2"=>"C",  "A3"=>"D")),
                            Pair("B", Dict("B1"=>"B1", "B2"=>"B2", "B3"=>"B3")),
                            Pair("C", Dict("C1"=>"C1", "C2"=>"C2", "C3"=>"C3")),
                            Pair("D", Dict("D1"=>"D1", "D2"=>"D2", "D3"=>"D3")),
                            ]),
                        Dict([
                            Pair("B1", 3),
                            Pair("B2", 12),
                            Pair("B3", 8),
                            Pair("C1", 2),
                            Pair("C2", 4),
                            Pair("C3", 6),
                            Pair("D1", 14),
                            Pair("D2", 5),
                            Pair("D3", 2),
                            ]));
    end
end

function actions(game::Figure52Game, state::String)
    return collect(keys(get(game.nodes, state, Dict())));
end

function result(game::Figure52Game, state::String, move::String)
    return game.nodes[state][move];
end

function utility(game::Figure52Game, state::String, player::String)
    if (player == "MAX")
        return game.utilities[state];
    else
        return -game.utilities[state];
    end
end

function terminal_test(game::Figure52Game, state::String)
    return !(state in ["A", "B", "C", "D"]);
end

function to_move(game::Figure52Game, state::String)
    return if_((state in ["B", "C", "D"]), "MIN", "MAX");
end

function minimax_max_value{T <: AbstractGame}(game::T, player::String, state::String)
    if (terminal_test(game, state))
        return utility(game, state, player)
    end
    local v::Float64 = -Inf64;
    v = reduce(max, vcat(v, collect(minimax_min_value(game, player, result(game, state, action))
                                    for action in actions(game, state))));
    return v;
end

function minimax_min_value{T <: AbstractGame}(game::T, player::String, state::String)
    if (terminal_test(game, state))
        return utility(game, state, player);
    end
    local v::Float64 = Inf64;
    v = reduce(min, vcat(v, collect(minimax_max_value(game, player, result(game, state, action))
                                    for action in actions(game, state))));
    return v;
end

"""
    minimax_decision(state, game)

Calculate the best move by searching through moves, all the way to the leaves (terminal states) (Fig 5.3).
"""
function minimax_decision{T <: AbstractGame}(state::String, game::T)
    local player = to_move(game, state);
    return argmax(actions(game, state),
                    (function(action::String,; relevant_game::AbstractGame=game, relevant_player::String=player, relevant_state::String=state)
                        return minimax_min_value(relevant_game, relevant_player, result(relevant_game, relevant_state, action));
                    end));
end

function alphabeta_full_search_max_value{T <: AbstractGame}(game::T, player::String, state::String, alpha::Number, beta::Number)
	if (terminal_test(game, state))
		return utility(game, state, player)
	end
	local v::Float64 = -Inf64;
	for action in actions(game, state)
		v = max(v, alphabeta_full_search_min_value(game, player, result(game, state, action), alpha, beta));
        if (v >= beta)
            return v;
        end
        alpha = max(alpha, v);
	end
	return v;
end

function alphabeta_full_search_min_value{T <: AbstractGame}(game::T, player::String, state::String, alpha::Number, beta::Number)
    if (terminal_test(game, state))
        return utility(game, state, player);
    end
    local v::Float64 = Inf64;
    for action in actions(game, state)
        v = min(v, alphabeta_full_search_max_value(game, player, result(game, state, action), alpha, beta));
        if (v <= alpha)
            return v;
        end
        beta = min(beta, v);
    end
    return v;
end

"""
    alphabeta_full_search(state, game)

Search the given game to find the best action using alpha-beta pruning (Fig 5.7).
"""
function alphabeta_full_search{T <: AbstractGame}(state::String, game::T)
	local player::String = to_move(game, state);
    return argmax(actions(game, state), 
                    (function(action::String,; relevant_game::AbstractGame=game, relevant_state::String=state, relevant_player::String=player)
                        return alphabeta_full_search_min_value(relevant_game, relevant_player, result(relevant_game, relevant_state, action), -Inf64, Inf64);
                    end));
end

function alphabeta_search_max_value{T <: AbstractGame}(game::T, player::String, cutoff_test_fn::Function, evaluation_fn::Function, state::String, alpha::Number, beta::Number, depth::Int64)
    if (cutoff_test_fn(state, depth))
        return evaluation_fn(state);
    end
    local v::Float64 = -Inf64;
    for action in actions(game, state)
        v = max(v, alphabeta_search_min_value(game, player, cutoff_test_fn, evaluation_fn, result(game, state, action), alpha, beta, depth + 1));
        if (v >= beta)
            return v;
        end
        alpha = max(alpha, v);
    end
    return v;
end

function alphabeta_search_min_value{T <: AbstractGame}(game::T, player::String, cutoff_test_fn::Function, evaluation_fn::Function, state::String, alpha::Number, beta::Number, depth::Int64)
    if (cutoff_test_fn(state, depth))
        return evaluation_fn(state);
    end
    local v::Float64 = Inf64;
    for action in actions(game, state)
        v = min(v, alphabeta_search_max_value(game, player, cutoff_test_fn, evaluation_fn, result(game, state, action), alpha, beta, depth + 1));
        if (v >= alpha)
            return v;
        end
        beta = min(alpha, v);
    end
    return v;
end

"""
    alphabeta_search(state, game)

Search the given game to find the best action using alpha-beta pruning. However, this function also uses a
cutoff test to cut off the search early and apply a heuristic evaluation function to turn nonterminal
states into terminal states.
"""
function alphabeta_search{T <: AbstractGame}(state::String, game::T; d::Int64=4, cutoff_test_fn::Union{Void, Function}=nothing, evaluation_fn::Union{Void, Function}=nothing)
    local player::String = to_move(game, state);
    if (typeof(cutoff_test_fn) <: Void)
        cutoff_test_fn = (function(state::String, depth::Int64; dvar::Int64=d, relevant_game::AbstractGame=game)
                            return ((depth > dvar) || terminal_test(relevant_game, state));
                        end);
    end
    if (typeof(evaluation_fn) <: Void)
        evaluation_fn = (function(state::String, ; relevant_game::AbstractGame=game, relevant_player::String=player)
                            return utility(relevant_game, state, relevant_player);
                        end);
    end
    return argmax(actions(game, state),
                    (function(action::String,; relevant_game::AbstractGame=game, relevant_state::String=state, relevant_player::String=player, cutoff_test::Function=cutoff_test_fn, eval_fn::Function=evaluation_fn)
                        return alphabeta_search_min_value(relevant_game, relevant_player, cutoff_test, eval_fn, result(relevant_game, relevant_state, action), -Inf64, Inf64, 0);
                    end));
end

