include("../aimajulia.jl");

using Base.Test;

using aimajulia;

#The following Planning tests are from the aima-python doctests

precondition = (map(expr, ["P(x)", "Q(y, z)"]), [expr("Q(x)")]);

effect = ([expr("Q(x)")], [expr("P(x)")]);

plan_action = PlanningAction(expr("A(x, y, z)"), precondition, effect);

arguments = map(expr, ["A", "B", "C"]);

@test (substitute(plan_action, expr("P(x, z, y)"), (arguments...)) == expr("P(A, C, B)"));

test_planning_kb = FirstOrderLogicKnowledgeBase(map(expr, ["P(A)", "Q(B, C)", "R(D)"]));

@test check_precondition(plan_action, test_planning_kb, (arguments...));

execute_action(plan_action, test_planning_kb, (arguments...));

# Found no valid substitutions!
@test (length(ask(test_planning_kb, expr("P(A)"))) == 0);

# Found valid substitutions!
@test (length(ask(test_planning_kb, expr("Q(A)"))) != 0);

# Found valid substitutions!
@test (length(ask(test_planning_kb, expr("Q(A)"))) != 0);

@test (!check_precondition(plan_action, test_planning_kb, (arguments...)));

air_cargo = air_cargo_pddl();

@test (goal_test(air_cargo) == false);

for action in map(expr, ("Load(C1, P1, SFO)", "Fly(P1, SFO, JFK)", "Unload(C1, P1, JFK)",
                        "Load(C2, P2, JFK)", "Fly(P2, JFK, SFO)", "Unload(C2, P2, SFO)"))
    execute_action(air_cargo, action);
end

@test goal_test(air_cargo);

air_cargo = air_cargo_pddl();

@test (goal_test(air_cargo) == false);

for action in map(expr, ("Load(C2, P2, JFK)", "Fly(P2, JFK, SFO)", "Unload(C2, P2, SFO)",
                        "Load(C1, P1, SFO)", "Fly(P1, SFO, JFK)", "Unload(C1, P1, JFK)"))
	execute_action(air_cargo, action);
end

@test goal_test(air_cargo);

spare_tire = spare_tire_pddl();

@test (goal_test(spare_tire) == false);

for action in map(expr, ("Remove(Flat, Axle)", "Remove(Spare, Trunk)", "PutOn(Spare, Axle)"))
    execute_action(spare_tire, action);
end

@test goal_test(spare_tire);

three_block_tower = three_block_tower_pddl();

@test (goal_test(three_block_tower) == false);

for action in map(expr, ("MoveToTable(C, A)", "Move(B, Table, C)", "Move(A, Table, B)"))
    execute_action(three_block_tower, action);
end

@test goal_test(three_block_tower);

have_cake_and_eat_cake_too = have_cake_and_eat_cake_too_pddl();

@test (goal_test(have_cake_and_eat_cake_too) == false);

for action in map(expr, ("Eat(Cake)", "Bake(Cake)"))
    execute_action(have_cake_and_eat_cake_too, action);
end

@test goal_test(have_cake_and_eat_cake_too);

spare_tire = spare_tire_pddl();

negated_kb = FirstOrderLogicKnowledgeBase([expr("At(Flat, Trunk)")]);

spare_tire_graph = PlanningGraph(spare_tire, negated_kb);

untouched_graph_levels_count = length(spare_tire_graph.levels);

expand_graph(spare_tire_graph);

@test (untouched_graph_levels_count == (length(spare_tire_graph.levels) - 1));

# Apply graphplan() to spare tire planning problem.

spare_tire = spare_tire_pddl();

negated_kb = FirstOrderLogicKnowledgeBase([expr("At(Flat, Trunk)")]);

spare_tire_gp = GraphPlanProblem(spare_tire, negated_kb);

@test (!(typeof(graphplan(spare_tire_gp, ([expr("At(Spare, Axle)"), expr("At(Flat, Ground)")], []))) <: Void));

doubles_tennis = doubles_tennis_pddl();

@test (goal_test(doubles_tennis) == false);

for action in map(expr, ["Go(A, LeftBaseLine, RightBaseLine)", "Hit(A, RightBaseLine, Ball)", "Go(A, RightBaseLine, LeftNet)"])
    execute_action(doubles_tennis, action);
end

@test goal_test(doubles_tennis);

# Create dictionary representation of possible refinements for "going to San Francisco airport HLA" (Fig. 11.4).
go_to_sfo_refinements_dict = Dict([Pair("HLA", ["Go(Home,SFO)", "Go(Home,SFO)", "Drive(Home, SFOLongTermParking)", "Shuttle(SFOLongTermParking, SFO)", "Taxi(Home, SFO)"]),
                                    Pair("steps", [["Drive(Home, SFOLongTermParking)", "Shuttle(SFOLongTermParking, SFO)"], ["Taxi(Home, SFO)"], [], [], []]),
                                    Pair("precondition_positive", [["At(Home), Have(Car)"], ["At(Home)"], ["At(Home)", "Have(Car)"], ["At(SFOLongTermParking)"], ["At(Home)"]]),
                                    Pair("precondition_negated", [[], [], [], [], []]),
                                    Pair("effect_add_list", [["At(SFO)"], ["At(SFO)"], ["At(SFOLongTermParking)"], ["At(SFO)"], ["At(SFO)"]]),
                                    Pair("effect_delete_list", [["At(Home)"], ["At(Home)"], ["At(Home)"], ["At(SFOLongTermParking)"], ["At(Home)"]])
                                    ]);

# Base.Test tests for refinements().
function test_refinement_goal_test(kb::FirstOrderLogicKnowledgeBase)
    return ask(kb, expr("At(SFO)"));
end

refinement_lib = Dict([Pair("HLA", ["Go(Home, SFO)", "Taxi(Home, SFO)"]),
                        Pair("steps", [["Taxi(Home, SFO)"], []]),
                        Pair("precondition_positive", [["At(Home)"], ["At(Home)"]]),
                        Pair("precondition_negated", [[],[]]),
                        Pair("effect_add_list", [["At(SFO)"],["At(SFO)"]]),
                        Pair("effect_delete_list", [["At(Home)"], ["At(Home)"]])]);
# Go to San Francisco airport high-level action schema
precondition_positive = Array{Expression, 1}([expr("At(Home)")]);
precondition_negated = Array{Expression, 1}([]);
effect_add_list = Array{Expression, 1}([expr("At(SFO)")]);
effect_delete_list = Array{Expression, 1}([expr("At(Home)")]);
go_sfo = PlanningHighLevelAction(expr("Go(Home, SFO)"),
                                (precondition_positive, precondition_negated),
                                (effect_add_list, effect_delete_list));
# Take Taxi to San Francisco airport high-level action schema
precondition_positive = Array{Expression, 1}([expr("At(Home)")]);
precondition_negated = Array{Expression, 1}([]);
effect_add_list = Array{Expression, 1}([expr("At(SFO)")]);
effect_delete_list = Array{Expression, 1}([expr("At(Home)")]);
taxi_sfo = PlanningHighLevelAction(expr("Go(Home, SFO)"),
                                    (precondition_positive, precondition_negated),
                                    (effect_add_list, effect_delete_list));
go_sfo_pddl = HighLevelPDDL(Array{Expression, 1}([expr("At(Home)")]), [go_sfo, taxi_sfo], test_refinement_goal_test);
result = refinements(go_sfo, go_sfo_pddl, refinement_lib);
@test (length(result) == 1);
@test (result[1].name == "Taxi");
@test (result[1].arguments == (expr("Home"), expr("SFO")));

job_shop_scheduling = job_shop_scheduling_pddl();

@test (goal_test(job_shop_scheduling) == false);

for i in reverse(1:2)
    for j in 1:3
        execute_action(job_shop_scheduling, job_shop_scheduling.jobs[i][j]);
    end
end

@test goal_test(job_shop_scheduling);

