
import Base.display;

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
    local v::FLoat64 = Inf64;
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
                    (function(actions::String)
                        return minimax_min_value(game, player, result(game, state, action));
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
    local v::FLoat64 = Inf64;
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
                    (function(action::String)
                        return alphabeta_full_search_min_value(game, player, result(game, state, action), -Inf64, Inf64);
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
        cutoff_test_fn = (function(state::String, depth::Int64; d::Int64=d, game::AbstractGame=game)
                            return ((depth > d) || terminal_test(game, state));
                        end);
    end
    if (typeof(evaluation_fn) <: Void)
        evaluation_fn = (function(state::String, ; game::AbstractGame=game, player::String=player)
                            return utility(game, state, player);
                        end);
    end
    return argmax(actions(game, state),
                    (function(action::String,; game::AbstractGame=game, state::String=state, cutoff_test::Function=cutoff_test_fn, eval_fn::Function=evaluation_fn)
                        return alphabeta_search_min_value(game, player, cutoff_test, eval_fn, result(game, state, action), -Inf64, Inf64, 0);
                    end));
end

