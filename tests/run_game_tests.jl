include("../aimajulia.jl");

using Base.Test;

using aimajulia;

#The following Game tests are from the aima-python doctest

@test minimax_decision("A", Figure52Game()) == "A1";

@test alphabeta_full_search("A", Figure52Game()) == "A1";

@test alphabeta_search("A", Figure52Game()) == "A1";

@test play_game(Figure52Game(), alphabeta_player, alphabeta_player) == 3;

#=

    The following tests may fail sometimes because the tests run on random behavior.

    However, the results of tests that fail does not imply something is wrong.

=#

function colorize_testv_doctest_results(result::Bool)
    if (result)
        print_with_color(:green, "Test Passed\n");
    else
        print_with_color(:red, "Test Failed\n");
    end
end

randf52_result = play_game(Figure52Game(), random_player, random_player);
colorize_testv_doctest_results(randf52_result == 6);
println("Expression: play_game(Figure52Game(), random_player, random_player) == 6");
println("Evaluated: ", randf52_result, " == 6");

randttt_result = play_game(TicTacToeGame(), random_player, random_player);
colorize_testv_doctest_results(randttt_result == 0);
println("Expression: play_game(TicTacToeGame(), random_player, random_player) == 0");
println("Evaluated: ", randttt_result, " == 0");
