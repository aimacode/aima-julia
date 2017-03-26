
import Base.display;

abstract AbstractGame;

#=

    Game is an abstract game that contains an initial state.

    Games have a corresponding utility function, terminal test, and transition model.

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
    for action in actions(game, state)
        v = max(v, minimax_min_value(result(game, state, action)));
    end
    return v;
end

function minimax_min_value{T <: AbstractGame}(game::T, player::String, state::String)
    if (terminal_test(game, state))
        return utility(game, state, player);
    end
    local v::FLoat64 = Inf64;
    for action in actions(game, state)
        v = min(v, minimax_max_value(result(game, state, action)));
    end
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
                        return minimax_decision_min_value(result(game, state, action));
                    end));
end

function alphabeta_full_search_max_value{T <: AbstractGame}(game::T, player::String, state::String, alpha::Number, beta::Number)
	if (terminal_test(game, state))
		return utility(game, state, player)
	end
	local v::Float64 = -Inf64;
	for action in actions(game, state)
		v = max(v, minimax_min_value(result(game, state, action)));
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
        v = min(v, minimax_max_value(result(game, state, action)));
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
	local player = to_move(game, state);
    return argmax(actions(game, state), 
                    (function(action::String)
                        return alphabeta_full_search_min_value(result(game, state, action), -Inf64, Inf64);
                    end));
end

