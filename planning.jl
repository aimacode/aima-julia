
export AbstractPDDL,
        PDDL, goal_test, execute_action,
        AbstractPlanningAction, PlanningAction,
        substitute, check_precondition,
        air_cargo_pddl, air_cargo_goal_test,
        spare_tire_pddl, spare_tire_goal_test,
        three_block_tower_pddl, three_block_tower_goal_test,
        have_cake_and_eat_cake_too_pddl, have_cake_and_eat_cake_too_goal_test,
        PlanningLevel,
        find_mutex_links, build_level_links, build_level_links_permute_arguments, perform_actions,
        PlanningGraph, expand_graph, non_mutex_goal_combinations, non_mutex_goals;

abstract AbstractPDDL;

abstract AbstractPlanningAction;

#=

    PlanningAction is an action schema defined by the action's name, preconditions, and effects.

    Preconditions and effects consists of either positive and negated literals.

=#
type PlanningAction <: AbstractPlanningAction
    name::String
    arguments::Tuple
    precondition_positive::Array{Expression, 1}
    precondition_negated::Array{Expression, 1}
    effect_add_list::Array{Expression, 1}
    effect_delete_list::Array{Expression, 1}

    function PlanningAction(action::Expression, precondition::Tuple{Vararg{Array{Expression, 1}, 2}}, effect::Tuple{Vararg{Array{Expression, 1}, 2}})
        return new(action.operator, action.arguments, precondition[1], precondition[2], effect[1], effect[2]);
    end
end

function substitute{T <: AbstractPlanningAction}(action::T, e::Expression, arguments::Tuple{Vararg{Expression}})
    local new_arguments::AbstractVector = collect(e.arguments);
    for (index_1, argument) in enumerate(e.arguments)
        for index_2 in 1:length(action.arguments)
            if (action.arguments[index_2] == argument)
                new_arguments[index_1] = arguments[index_2];
            end
        end
    end
    return Expression(e.operator, Tuple((new_arguments...)));
end

function check_precondition{T1 <: AbstractPlanningAction, T2 <: AbstractKnowledgeBase}(action::T1, kb::T2, arguments::Tuple)
    # Check for positive clauses.
    for clause in action.precondition_positive
        if (!(substitute(action, clause, arguments) in kb.clauses))
            return false;
        end
    end
    # Check for negated clauses.
    for clause in action.precondition_negated
        if (substitute(action, clause, arguments) in kb.clauses)
            return false;
        end
    end
    return true;
end

function execute_action{T1 <: AbstractPlanningAction, T2 <: AbstractKnowledgeBase}(action::T1, kb::T2, arguments::Tuple)
    if (!(check_precondition(action, kb, arguments)))
        error(@sprintf("execute_action(): Action \"%s\" preconditions are not satisfied!", action.name));
    end
    # Retract negated literals to knowledge base 'kb'.
    for clause in action.effect_delete_list
        retract(kb, substitute(action, clause, arguments));
    end
    # Add positive literals to knowledge base 'kb'.
    for clause in action.effect_add_list
        tell(kb, substitute(action, clause, arguments));
    end
    nothing;
end

#=

    The Planning Domain Definition Language (PDDL) is used to define a search problem.

    The states (starting from the initial state) are represented as the conjunction of

    the statements in 'kb' (a FirstOrderLogicKnowledgeBase). The actions are described

    by 'actions' (an array of action schemas). The 'goal_test' is a function that checks

    if the current state of the problem is at the goal state.

=#
type PDDL <: AbstractPDDL
    kb::FirstOrderLogicKnowledgeBase
    actions::Array{PlanningAction, 1}
    goal_test::Function

    function PDDL(initial_state::Array{Expression, 1}, actions::Array{PlanningAction, 1}, goal_test::Function)
        return new(FirstOrderLogicKnowledgeBase(initial_state), actions, goal_test);
    end
end

function goal_test{T <: AbstractPDDL}(plan::T)
    return plan.goal_test(plan.kb);
end

function execute_action{T <: AbstractPDDL}(plan::T, action::Expression)
    local action_name::String = action.operator;
    local arguments::Tuple = action.arguments;
    local relevant_actions::AbstractVector = collect(a for a in plan.actions if (a.name == action_name));
    if (length(relevant_actions) == 0)
        error(@sprintf("execute_action(): Action \"%s\" not found!", action_name));
    else
        local first_relevant_action::PlanningAction = relevant_actions[1];
        if (!check_precondition(first_relevant_action, plan.kb, arguments))
            error(@sprintf("execute_action(): Action \"%s\" preconditions are not satisfied!", repr(action)));
        else
            execute_action(first_relevant_action, plan.kb, arguments);
        end
    end
    nothing;
end

function air_cargo_goal_test(kb::FirstOrderLogicKnowledgeBase)
    return all((function(ans)
                    if (typeof(ans) <: Bool)
                        return ans;
                    else
                        if (length(ans) == 0)   # length of Tuple
                            return false;
                        else
                            return true;
                        end
                    end
                end),
                collect(ask(kb, q) for q in (expr("At(C1, JFK)"), expr("At(C2, SFO)"))));
end

"""
    air_cargo_pddl()

Return a PDDL representing the air cargo transportation planning problem (Fig. 10.1).
"""
function air_cargo_pddl()
    local initial::Array{Expression, 1} = map(expr, ["At(C1, SFO)",
                                                "At(C2, JFK)",
                                                "At(P1, SFO)",
                                                "At(P2, JFK)",
                                                "Cargo(C1)",
                                                "Cargo(C2)",
                                                "Plane(P1)",
                                                "Plane(P2)",
                                                "Airport(JFK)",
                                                "Airport(SFO)"]);
    # Load Action Schema
    local precondition_positive::Array{Expression, 1} = map(expr, ["At(c, a)",
                                                            "At(p, a)",
                                                            "Cargo(c)",
                                                            "Plane(p)",
                                                            "Airport(a)"]);
    local precondition_negated::Array{Expression, 1} = [];
    local effect_add_list::Array{Expression, 1} = [expr("In(c, p)")];
    local effect_delete_list::Array{Expression, 1} = [expr("At(c, a)")];
    local load::PlanningAction = PlanningAction(expr("Load(c, p, a)"),
                                                (precondition_positive, precondition_negated),
                                                (effect_add_list, effect_delete_list));
    # Unload Action Schema
    precondition_positive = map(expr, ["In(c, p)", "At(p, a)", "Cargo(c)", "Plane(p)", "Airport(a)"]);
    precondition_negated = [];
    effect_add_list = [expr("At(c, a)")];
    effect_delete_list = [expr("In(c, p)")];
    local unload::PlanningAction = PlanningAction(expr("Unload(c, p, a)"),
                                                (precondition_positive, precondition_negated),
                                                (effect_add_list, effect_delete_list));
    # Fly Action Schema
    precondition_positive = map(expr, ["At(p, f)", "Plane(p)", "Airport(f)", "Airport(to)"]);
    precondition_negated = [];
    effect_add_list = [expr("At(p, to)")];
    effect_delete_list = [expr("At(p, f)")];
    local fly::PlanningAction = PlanningAction(expr("Fly(p, f, to)"),
                                                (precondition_positive, precondition_negated),
                                                (effect_add_list, effect_delete_list));
    return PDDL(initial, [load, unload, fly], air_cargo_goal_test);
end

function spare_tire_goal_test(kb::FirstOrderLogicKnowledgeBase)
    return all((function(ans)
                    if (typeof(ans) <: Bool)
                        return ans;
                    else
                        if (length(ans) == 0)   # length of Tuple
                            return false;
                        else
                            return true;
                        end
                    end
                end),
                collect(ask(kb, q) for q in (expr("At(Spare, Axle)"),)));
end

"""
    spare_tire_pddl()

Return a PDDL representing the spare tire planning problem (Fig. 10.2).
"""
function spare_tire_pddl()
    local initial::Array{Expression, 1} = map(expr, ["Tire(Flat)",
                                                    "Tire(Spare)",
                                                    "At(Flat, Axle)",
                                                    "At(Spare, Trunk)"]);
    # Remove Action Schema
    local precondition_positive::Array{Expression, 1} = [expr("At(obj, loc)")];
    local precondition_negated::Array{Expression, 1} = [];
    local effect_add_list::Array{Expression, 1} = [expr("At(obj, Ground)")];
    local effect_delete_list::Array{Expression, 1} = [expr("At(obj, loc)")];
    local remove::PlanningAction = PlanningAction(expr("Remove(obj, loc)"),
                                                (precondition_positive, precondition_negated),
                                                (effect_add_list, effect_delete_list));
    # PutOn Action Schema
    precondition_positive = map(expr, ["Tire(t)", "At(t, Ground)"]);
    precondition_negated = [expr("At(Flat, Axle)")];
    effect_add_list = [expr("At(t, Axle)")];
    effect_delete_list = [expr("At(t, Ground)")];
    local put_on::PlanningAction = PlanningAction(expr("PutOn(t, Axle)"),
                                                    (precondition_positive, precondition_negated),
                                                    (effect_add_list, effect_delete_list));
    # LeaveOvernight Action Schema
    precondition_positive = [];
    precondition_negated = [];
    effect_add_list = [];
    effect_delete_list = map(expr, ["At(Spare, Ground)", "At(Spare, Axle)", "At(Spare, Trunk)",
                                    "At(Flat, Ground)", "At(Flat, Axle)", "At(Flat, Trunk)"]);
    local leave_overnight::PlanningAction = PlanningAction(expr("LeaveOvernight"),
                                                            (precondition_positive, precondition_negated),
                                                            (effect_add_list, effect_delete_list));
    return PDDL(initial, [remove, put_on, leave_overnight], spare_tire_goal_test);
end

function three_block_tower_goal_test(kb::FirstOrderLogicKnowledgeBase)
    return all((function(ans)
                    if (typeof(ans) <: Bool)
                        return ans;
                    else
                        if (length(ans) == 0)   #length of Tuple
                            return false;
                        else
                            return true;
                        end
                    end
                end),
                collect(ask(kb, q) for q in (expr("On(A, B)"), expr("On(B, C)"))));
end

"""
    three_block_tower_pddl()

Return a PDDL representing the building of a three-block tower planning problem (Fig. 10.3).
"""
function three_block_tower_pddl()
    local initial::Array{Expression, 1} = map(expr, ["On(A, Table)",
                                                    "On(B, Table)",
                                                    "On(C, A)",
                                                    "Block(A)",
                                                    "Block(B)",
                                                    "Block(C)",
                                                    "Clear(B)",
                                                    "Clear(C)"]);
    # Move Action Schema
    local precondition_positive::Array{Expression, 1} = map(expr, ["On(b, x)", "Clear(b)", "Clear(y)", "Block(b)", "Block(y)"]);
    local precondition_negated::Array{Expression, 1} = [];
    local effect_add_list::Array{Expression, 1} = [expr("On(b, y)"), expr("Clear(x)")];
    local effect_delete_list::Array{Expression, 1} = [expr("On(b, x)"), expr("Clear(y)")];
    local move::PlanningAction = PlanningAction(expr("Move(b, x, y)"),
                                                (precondition_positive, precondition_negated),
                                                (effect_add_list, effect_delete_list));
    # MoveToTable Action Schema
    precondition_positive = map(expr, ["On(b, x)", "Clear(b)", "Block(b)"]);
    precondition_negated = [];
    effect_add_list = [expr("On(b, Table)"), expr("Clear(x)")];
    effect_delete_list = [expr("On(b, x)")];
    local move_to_table::PlanningAction = PlanningAction(expr("MoveToTable(b, x)"),
                                                        (precondition_positive, precondition_negated),
                                                        (effect_add_list, effect_delete_list));
    return PDDL(initial, [move, move_to_table], three_block_tower_goal_test);
end

function have_cake_and_eat_cake_too_goal_test(kb::FirstOrderLogicKnowledgeBase)
    return all((function(ans)
                    if (typeof(ans) <: Bool)
                        return ans;
                    else
                        if (length(ans) == 0)   # length of Tuple
                            return false;
                        else
                            return true;
                        end
                    end
                end),
                collect(ask(kb, q) for q in (expr("Have(Cake)"), expr("Eaten(Cake)"))));
end

"""
    have_cake_and_eat_cake_too_pddl()

Return a PDDL representing the 'have cake and eat cake too' planning problem (Fig. 10.7).
"""
function have_cake_and_eat_cake_too_pddl()
    local initial::Array{Expression, 1} = [expr("Have(Cake)")];
    # Eat Cake Action Schema
    local precondition_positive::Array{Expression, 1} = [expr("Have(Cake)")];
    local precondition_negated::Array{Expression, 1} = [];
    local effect_add_list::Array{Expression, 1} = [expr("Eaten(Cake)")];
    local effect_delete_list::Array{Expression, 1} = [expr("Have(Cake)")];
    local eat_cake::PlanningAction = PlanningAction(expr("Eat(Cake)"),
                                                    (precondition_positive, precondition_negated),
                                                    (effect_add_list, effect_delete_list));
    # Bake Cake Action Schema
    precondition_positive = [];
    precondition_negated = [expr("Have(Cake)")];
    effect_add_list = [expr("Have(Cake)")];
    effect_delete_list = [];
    local bake_cake::PlanningAction = PlanningAction(expr("Bake(Cake)"),
                                                    (precondition_positive, precondition_negated),
                                                    (effect_add_list, effect_delete_list));
    return PDDL(initial, [eat_cake, bake_cake], have_cake_and_eat_cake_too_goal_test);
end

type PlanningLevel
    positive_kb::FirstOrderLogicKnowledgeBase
    current_state_positive::Array{Expression, 1}    #current state of the planning problem
    current_state_negated::Array{Expression, 1}     #current state of the planning problem
    current_action_links_positive::Dict             #current actions to current state link
    current_action_links_negated::Dict              #current actions to current state link
    current_state_links_positive::Dict              #current state to action link
    current_state_links_negated::Dict               #current state to action link
    next_action_links::Dict                         #current action to next state link
    next_state_links_positive::Dict                 #next state to current action link
    next_state_links_negated::Dict                  #next state to current action link
    mutex_links::Array{Set, 1}                      #each mutex relation is a Set of 2 actions/literals

    function PlanningLevel(p_kb::FirstOrderLogicKnowledgeBase, n_kb::FirstOrderLogicKnowledgeBase)
        return new(p_kb, p_kb.clauses, n_kb.clauses, Dict(), Dict(), Dict(), Dict(), Dict(), Dict(), Dict(), []);
    end
end

function find_mutex_links(level::PlanningLevel)
    # Inconsistent effects condition between 2 action schemas at a given level
    for positve_effect in level.next_state_links_positive
        negated_effect = positive_effect;
        if (haskey(level.next_state_links_negated, negated_effect))
            for a in level.next_state_links_positive[positive_effect]
                for b in level.next_state_links_negated[negated_effect]
                    if (!(Set([a, b]) in level.mutex_links))
                        push!(level.mutex_links, Set([a, b]));
                    end
                end
            end
        end
    end
    # Inference condition between 2 action schemas at a given level
    for positive_precondition in level.current_state_links_positive
        negated_effect = positive_precondition;
        if (haskey(level.next_state_links_negated, negated_effect))
            for a in level.current_state_links_positive[positive_precondition]
                for b in level.next_state_links_negated[negated_effect]
                    if (!(Set([a, b]) in level.mutex_links))
                        push!(level.mutex_links, Set([a, b]));
                    end
                end
            end
        end
    end
    for negated_precondition in level.current_state_links_negated
        positive_effect = negated_precondition;
        if (haskey(level.next_state_links_positive, positive_effect))
            for a in level.next_state_links_positive[positive_effect]
                for b in level.current_state_links_negated[negated_precondition]
                    if (!(Set([a, b]) in level.mutex_links))
                        push!(level.mutex_links, Set([a, b]));
                    end
                end
            end
        end
    end
    # Competing needs condition between 2 action schemas
    for positive_precondition in level. current_state_links_positive
        negated_precondition = positive_precondition;
        if (haskey(level.current_state_links_negated, negated_precondition))
            for a in level.current_state_links_positive[positive_precondition]
                for b in level.current_state_links_negated[negated_precondition]
                    if (!(Set([a, b]) in level.mutex_links))
                        push!(level.mutex_links, Set([a, b]));
                    end
                end
            end
        end
    end
    # Inconsistent support condition
    local state_mutex_links::AbstractVector = [];
    for pair in level.mutex_links
        collected_pair::AbstractVector = collect(pair);
        next_state_1 = level.next_action_links[collected_pair[1]];
        if (length(sorted_pair) == 2)
            next_state_2 = level.next_action_links[collected_pair[2]];
        else
            next_state_2 = level.next_action_links[collected_pair[1]];
        end
        if ((length(next_state_1) == 1) && (length(next_state_2) == 1))
            push!(state_mutex_links, Set([next_state_1[1], next_state_2[1]]));
        end
    end
    level.mutex_links = vcat(level.mutex_links, state_mutex_links);
    nothing;
end

function build_level_links_permute_arguments(depth::Int64, objects::AbstractVector, current_permutation::Tuple, permutations_array::AbstractVector)
    if (depth == 0)
        push!(permutations_array, current_permutation);
    elseif (depth < 0)
        error("build_level_links_permute_arguments(): Found negative depth!");
    else
        for (i, item) in enumerate(objects)
            build_level_links_permute_arguments((depth - 1),
                                                Tuple((objects[1:(i - 1)]..., objects[(i + 1):end]...)),
                                                Tuple((current_permutation..., item)),
                                                permutations_array)
        end
    end
end

function build_level_links_permute_arguments(depth::Int64, objects::Tuple, current_permutation::Tuple, permutations_array::AbstractVector)
    if (depth == 0)
        push!(permutations_array, current_permutation);
    elseif (depth < 0)
        error("build_level_links_permute_arguments(): Found negative depth!");
    else
        for (i, item) in enumerate(objects)
            build_level_links_permute_arguments((depth - 1),
                                                Tuple((objects[1:(i - 1)]..., objects[(i + 1):end]...)),
                                                Tuple((current_permutation..., item)),
                                                permutations_array)
        end
    end
end

function build_level_links(level::PlanningLevel, actions::AbstractVector, objects::Set)
    # Create persistence actions for positive states
    for clause in level.current_state_positive
        level.current_action_links_positive[Expression("Persistence", clause)] = [clause];
        level.next_action_links[Expression("Persistence", clause)] = [clause];
        level.current_state_links_positive[clause] = [Expression("Persistence", clause)];
        level.next_state_links_positive[clause] = [Expression("Persistence", clause)];
    end
    # Create persistence actions for negated states
    for clause in level.current_state_negated
        not_expression = Expression("not"*clause.operator, clause.arguments);
        level.current_action_links_negated[Expression("Persistence", not_expression)] = [clause];
        level.next_action_links[Expression("Persistence", not_expression)] = [clause];
        level.current_state_links_negated[clause] = [Expression("Persistence", not_expression)];
        level.next_state_links_negated[clause] = [Expression("Persistence", not_expression)];
    end
    # Recursively collect num_arg depth, collecting a Tuple of Tuples
    for action in actions
        local num_arguments::Int64 = length(action.arguments);
        local possible_arguments::AbstractVector = [];
        build_level_links_permute_arguments(num_arguments, collect(objects), (), possible_arguments);
        for argument in possible_arguments
            if (check_precondition(action, level.positive_kb, argument))
                for (number, symbol) in enumerate(action.arguments)
                    if (!islower(symbol.operator))
                        argument = Tuple((argument[1:(number - 1)]..., symbol, argument[(number + 1):end]));
                    end
                end
                local new_action::Expression = substitute(action, Expression(action.name, action.arguments), argument);
                level.current_action_links_positive[new_action] = [];
                level.current_action_links_negated[new_action] = [];
                local new_clause::Expression;
                for clause in action.precondition_positive
                    new_clause = substitute(action, clause, argument);
                    push!(level.current_action_links_positive[new_action], new_clause);
                    if (haskey(level.current_state_links_positive, new_clause))
                        push!(level.current_state_links_positive[new_clause], new_action);
                    else
                        level.current_state_links_positive[new_clause] = [new_action];
                    end
                end
                for clause in action.precondition_negated
                    new_clause = substitute(action, clause, argument);
                    push!(level.current_action_links_negated[new_action], new_clause);
                    if (haskey(level.current_state_links_negated, new_clause))
                        push!(level.current_state_links_negated[new_clause], new_action);
                    else
                        level.current_state_links_negated[new_clause] = [new_action];
                    end
                end
                level.next_action_links[new_action] = [];
                for clause in action.effect_add_list
                    new_clause = substitute(action, clause, argument);
                    push!(level.next_action_links[new_action], new_clause);
                    if (haskey(level.next_state_links_positive, new_clause))
                        push!(level.next_state_links_positive[new_clause], new_action);
                    else
                        level.next_state_links_positive[new_clause] = [new_action];
                    end
                end
                for clause in action.effect_delete_list
                    new_clause = substitute(action, clause, argument);
                    push!(level.next_action_links[new_action], new_clause);
                    if (haskey(level.next_state_links_negated, new_clause))
                        push!(level.next_state_links_negated[new_clause], new_action);
                    else
                        level.next_state_links_negated[new_clause] = [new_action];
                    end
                end
            end
        end
    end
    nothing;
end

function perform_actions(level::PlanningLevel)
    local new_kb_positive::FirstORderLogicKnowledgeBase = FirstOrderLogicKnowledgeBase(collect(Set(collect(keys(level.next_state_links_positive)))));
    local new_kb_negated::FirstORderLogicKnowledgeBase = FirstOrderLogicKnowledgeBase(collect(Set(collect(keys(level.next_state_links_negated)))));
    return PlanningLevel(new_kb_positive, new_kb_negated);
end

type PlanningGraph
    pddl::AbstractPDDL
    levels::Array{PlanningLevel, 1}
    objects::Set{Expression}

    function PlanningGraph{T <: AbstractPDDL}(pddl::T, n_kb::FirstOrderLogicKnowledgeBase)
        return new(pddl, [PlanningLevel(pddl.kb, n_kb)], Set(arg for clause in vcat(pddl.kb.clauses, n_kb.clauses) for arg in clause.arguments));
    end
end

function expand_graph(pg::PlanningGraph)
    local last_level = pg.levels[length(pg.levels)];
    build_level_links(last_level, pg.pddl.actions, pg.objects);
    find_mutex_links(last_level);
    push!(pg.levels, perform_actions(last_level));
    nothing;
end

function non_mutex_goal_combinations(goals::AbstractVector) #ordered permutations of length 2
    local combinations::AbstractVector = [];
    for (i, a) in enumerate(goals)
        for b in goals[(i + 1):end]
            push!(combinations, (a, b));
        end
    end
    return combinations;
end

function non_mutex_goals(pg::PlanningGraph, goals::AbstractVector, index::Int64)
    local goal_combinations::AbstractVector = non_mutex_goal_combinations(goals);
    for goal in goal_combinations
        if (Set(collect(goal)) in pg.levels[index].mutex)
            return false;
        end
    end
    return true;
end

