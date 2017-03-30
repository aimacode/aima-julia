
import Base.display;

export AbstractGame, Figure52Game, TicTacToeGame, ConnectFourGame,
        TicTacToeState, ConnectFourState,
        minimax_decision, alphabeta_full_search, alphabeta_search,
        display,
        random_player, alphabeta_player, play_game;

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

type TicTacToeState
    turn::String
    utility::Int64
    board::Dict
    moves::AbstractVector

    function TicTacToeState(turn::String, utility::Int64, board::Dict, moves::AbstractVector)
        return new(turn, utility, board, moves);
    end
end

#=

    TicTacToeGame is a AbstractGame implementation of the Tic-tac-toe game.

=#
type TicTacToeGame <: AbstractGame
    initial::TicTacToeState
    h::Int64
    v::Int64
    k::Int64

    function TicTacToeGame(initial::TicTacToeState)
        return new(initial, 3, 3, 3);
    end

    function TicTacToeGame()
        return new(TicTacToeState("X", 0, Dict(), collect((x, y) for x in 1:3 for y in 1:3)), 3, 3, 3);
    end
end

function actions(game::TicTacToeGame, state::TicTacToeState)
    return state.moves;
end

function result(game::TicTacToeGame, state::TicTacToeState, move::Tuple{Signed, Signed})
    if (!(move in state.moves))
        return state;
    end
    local board::Dict = copy(state.board);
    board[move] = state.turn;
    local moves::Array{eltype(state.moves), 1} = collect(state.moves);
    for (i, element) in enumerate(moves)
        if (element == move)
            deleteat!(moves, i);
            break;
        end
    end
    return TicTacToeState(if_((state.turn == "X"), "O", "X"), compute_utility(game, board, move, state.turn), board, moves);
end

function utility(game::TicTacToeGame, state::TicTacToeState, player::String)
    return if_((player == "X"), state.utility, -state.utility);
end

function terminal_test(game::TicTacToeGame, state::TicTacToeState)
    return ((state.utility != 0) || (length(state.moves) == 0));
end

function to_move(game::TicTacToeGame, state::TicTacToeState)
    return state.turn;
end

function display(game::TicTacToeGame, state::TicTacToeState)
    for x in 1:game.h
        for y in 1:game.v
            print(get(state.board, (x, y), "."));
        end
        println();
    end
end

function compute_utility{T <: Dict}(game::TicTacToeGame, board::T, move::Tuple{Signed, Signed}, player::String)
    if (k_in_row(game, board, move, player, (0, 1)) ||
        k_in_row(game, board, move, player, (1, 0)) ||
        k_in_row(game, board, move, player, (1, -1)) ||
        k_in_row(game, board, move, player, (1, 1)))
        return if_((player == "X"), 1, -1);
    else
        return 0;
    end
end

function k_in_row(game::TicTacToeGame, board::Dict, move::Tuple{Signed, Signed}, player::String, delta::Tuple{Signed, Signed})
    local delta_x::Int64 = Int64(getindex(delta, 1));
    local delta_y::Int64 = Int64(getindex(delta, 2));
    local x::Int64 = Int64(getindex(move, 1));
    local y::Int64 = Int64(getindex(move, 2));
    local n::Int64 = Int64(0);
    while (get(board, (x,y), nothing) == player)
        n = n + 1;
        x = x + delta_x;
        y = y + delta_y;
    end
    x = Int64(getindex(move, 1));
    y = Int64(getindex(move, 2));
    while (get(board, (x,y), nothing) == player)
        n = n + 1;
        x = x - delta_x;
        y = y - delta_y;
    end
    n = n - 1;  #remove the duplicate check on get(board, move, nothing)
    return n >= game.k;
end

typealias ConnectFourState TicTacToeState;

#=

    ConnectFourGame is a AbstractGame implementation of the Connect Four game.

=#
type ConnectFourGame <: AbstractGame
    initial::ConnectFourState
    h::Int64
    v::Int64
    k::Int64

    function ConnectFourGame(initial::ConnectFourState)
        return new(initial, 3, 3, 3);
    end

    function ConnectFourGame()
        return new(ConnectFourState("X", 0, Dict(), collect((x, y) for x in 1:7 for y in 1:6)), 7, 6, 4);
    end
end

function actions(game::ConnectFourGame, state::ConnectFourState)
    return collect((x,y) for (x, y) in state.moves if ((y == 0) || ((x, y - 1) in state.board)));
end

function result(game::ConnectFourGame, state::ConnectFourState, move::Tuple{Signed, Signed})
    if (!(move in state.moves))
        return state;
    end
    local board::Dict = copy(state.board);
    board[move] = state.turn;
    local moves::Array{eltype(state.moves), 1} = collect(state.moves);
    for (i, element) in enumerate(moves)
        if (element == move)
            deleteat!(moves, i);
            break;
        end
    end
    return ConnectFourState(if_((state.turn == "X"), "O", "X"), compute_utility(game, board, move, state.turn), board, moves);
end

function utility(game::ConnectFourGame, state::ConnectFourState, player::String)
    return if_((player == "X"), state.utility, -state.utility);
end

function terminal_test(game::ConnectFourGame, state::ConnectFourState)
    return ((state.utility != 0) || (length(state.moves) == 0));
end

function to_move(game::ConnectFourGame, state::ConnectFourState)
    return state.turn;
end

function display(game::ConnectFourGame, state::ConnectFourState)
    for x in 1:game.h
        for y in 1:game.v
            print(get(state.board, (x, y), "."));
        end
        println();
    end
end

function compute_utility{T <: Dict}(game::ConnectFourGame, board::T, move::Tuple{Signed, Signed}, player::String)
    if (k_in_row(game, board, move, player, (0, 1)) ||
        k_in_row(game, board, move, player, (1, 0)) ||
        k_in_row(game, board, move, player, (1, -1)) ||
        k_in_row(game, board, move, player, (1, 1)))
        return if_((player == "X"), 1, -1);
    else
        return 0;
    end
end

function k_in_row(game::ConnectFourGame, board::Dict, move::Tuple{Signed, Signed}, player::String, delta::Tuple{Signed, Signed})
    local delta_x::Int64 = Int64(getindex(delta, 1));
    local delta_y::Int64 = Int64(getindex(delta, 2));
    local x::Int64 = Int64(getindex(move, 1));
    local y::Int64 = Int64(getindex(move, 2));
    local n::Int64 = Int64(0);
    while (get(board, (x,y), nothing) == player)
        n = n + 1;
        x = x + delta_x;
        y = y + delta_y;
    end
    x = Int64(getindex(move, 1));
    y = Int64(getindex(move, 2));
    while (get(board, (x,y), nothing) == player)
        n = n + 1;
        x = x - delta_x;
        y = y - delta_y;
    end
    n = n - 1;  #remove the duplicate check on get(board, move, nothing)
    return n >= game.k;
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

function alphabeta_search_max_value{T <: AbstractGame}(game::T, player::String, cutoff_test_fn::Function, evaluation_fn::Function, state::TicTacToeState, alpha::Number, beta::Number, depth::Int64)
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

function alphabeta_search_min_value{T <: AbstractGame}(game::T, player::String, cutoff_test_fn::Function, evaluation_fn::Function, state::TicTacToeState, alpha::Number, beta::Number, depth::Int64)
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

function alphabeta_search{T <: AbstractGame}(state::TicTacToeState, game::T; d::Int64=4, cutoff_test_fn::Union{Void, Function}=nothing, evaluation_fn::Union{Void, Function}=nothing)
    local player::String = to_move(game, state);
    if (typeof(cutoff_test_fn) <: Void)
        cutoff_test_fn = (function(state::TicTacToeState, depth::Int64; dvar::Int64=d, relevant_game::AbstractGame=game)
                            return ((depth > dvar) || terminal_test(relevant_game, state));
                        end);
    end
    if (typeof(evaluation_fn) <: Void)
        evaluation_fn = (function(state::TicTacToeState, ; relevant_game::AbstractGame=game, relevant_player::String=player)
                            return utility(relevant_game, state, relevant_player);
                        end);
    end
    return argmax(actions(game, state),
                    (function(action::Tuple{Signed, Signed},; relevant_game::AbstractGame=game, relevant_state::TicTacToeState=state, relevant_player::String=player, cutoff_test::Function=cutoff_test_fn, eval_fn::Function=evaluation_fn)
                        return alphabeta_search_min_value(relevant_game, relevant_player, cutoff_test, eval_fn, result(relevant_game, relevant_state, action), -Inf64, Inf64, 0);
                    end));
end

function random_player{T <: AbstractGame}(game::T, state::String)
    return rand(RandomDeviceInstance, actions(game, state));
end

function random_player{T <: AbstractGame}(game::T, state::TicTacToeState)
    return rand(RandomDeviceInstance, actions(game, state));
end

function alphabeta_player{T <: AbstractGame}(game::T, state::String)
    return alphabeta_search(state, game);
end

function alphabeta_player{T <: AbstractGame}(game::T, state::TicTacToeState)
    return alphabeta_search(state, game);
end

function play_game{T <: AbstractGame}(game::T, players::Vararg{Function})
    state = game.initial;
    while (true)
        for player in players
            move = player(game, state);
            state = result(game, state, move);
            if (terminal_test(game, state))
                return utility(game, state, to_move(game, game.initial));
            end
        end
    end
end

