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

