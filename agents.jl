import Base: run;

export AgentProgram,
        TableDrivenAgentProgram,
        ReflexVacuumAgentProgram,
        ModelBasedVacuumAgentProgram,
        RandomAgentProgram,
        Rule, SimpleReflexAgentProgram,
        ModelBasedReflexAgentProgram,
        EnvironmentObject, EnvironmentAgent,
        Agent, Wumpus, Explorer,
        isAlive, setAlive,
        Obstacle, Wall,
        Dirt, Gold, Pit, Arrow,
        execute, rule_match, interpret_input, update_state,
        TableDrivenVacuumAgent,
        ReflexVacuumAgent,
        ModelBasedVacuumAgent,
        RandomVacuumAgent,
        Environment, TwoDimensionalEnvironment, XYEnvironment,
        VacuumEnvironment, TrivialVacuumEnvironment, WumpusEnvironment,
        percept, objects_near, get_objects_at, some_objects_at, is_done, step,
        run, exogenous_change, default_location, environment_objects, execute_action,
        add_object, delete_object, add_walls, move_to,
        run_once,
        test_agent, compare_agents;


#=

    Define a global execute() function to be implemented for each respective
    AgentProgram DataType implementation.
        
=#

function execute{T <: AgentProgram}(ap::T, p::Tuple{Any, Any})      #implement functionality later
    #comment the following line to reduce verbosity
    #println("execute() is not implemented yet for ", typeof(ap), "!");
    nothing;
end

type TableDrivenAgentProgram <: AgentProgram
    isTracing::Bool
    percepts::Array{Tuple{Any, Any}, 1}
    table::Dict{Any, Any}

    function TableDrivenAgentProgram(;table_dict::Union{Void, Dict{Any, Any}}=nothing, trace::Bool=false)
        if (table_dict == C_NULL)           #no table given, create empty dictionary
            return new(Bool(trace), Array{Tuple{Any, Any}, 1}(), Dict{Any, Any}());
        else
            return new(Bool(trace), Array{Tuple{Any, Any}, 1}(), table_dict);
        end
    end
end

type ReflexVacuumAgentProgram <: AgentProgram
    isTracing::Bool

    function ReflexVacuumAgentProgram(;trace::Bool=false)
        return new(Bool(trace));
    end
end

type ModelBasedVacuumAgentProgram <: AgentProgram
    isTracing::Bool
    model::Dict{Any, Any}

    function ModelBasedVacuumAgentProgram(;trace::Bool=false, model::Union{Void, Dict{Any, Any}}=nothing)
        if (typeof(model) <: Dict{Any, Any})
            new_ap = new(Bool(trace));
            new_ap.model = deepcopy(model);
            return new_ap;
        else
            return new(Bool(trace), Dict{Any, Any}());
        end
    end
end

type RandomAgentProgram <: AgentProgram
    isTracing::Bool
    actions::Array{String, 1}

    function RandomAgentProgram(actions::Array{String, 1}; trace::Bool=false)
        return new(Bool(trace), deepcopy(actions));
    end
end

type Rule   #condition-action rules
    condition::String
    action::String

    function Rule(cond::String, action::String)
        return new(cond, action);
    end
end

type SimpleReflexAgentProgram <: AgentProgram
    isTracing::Bool
    rules::Array{Rule, 1}

    function SimpleReflexAgentProgram(rules_array::Array{Rule, 1};trace::Bool=false)
        srap = new(Bool(trace));
        srap.rules = deepcopy(rules_array);
        return rules_array;
    end
end

type ModelBasedReflexAgentProgram <: AgentProgram
    isTracing::Bool
    state::Dict{Any, Any}
    model::Dict{Any, Any}
    rules::Array{Rule, 1}
    action::String          #most recent action, initialized to empty string ""

    function ModelBasedReflexAgentProgram(state::Dict{Any, Any}, model::Dict{Any, Any}, rules::Array{Rule, 1}; trace::Bool=false)
        mbrap = new(Bool(trace));
        mbrap.state = deepcopy(state);
        mbrap.model = deepcopy(model);
        mbrap.rules = deepcopy(rules);
        return mbrap;
    end
end

#=

    Agents can interact with the environment through percepts and actions.

=#

abstract EnvironmentObject;         #declare EnvironmentObject as a supertype for EnvironmentObject implementations

#the EnvironmentAgent implementations exist in the environment like other EnvironmentObjects such as Gold or Dirt
abstract EnvironmentAgent <: EnvironmentObject;

type Agent <: EnvironmentAgent
    alive::Bool
    performance::Float64
    bump::Bool
    heading::Tuple{Any, Any}
    holding::Array{Any, 1}
    program::AgentProgram
    location::Tuple{Any, Any}       #initialized when adding agent to environment

    function Agent()
        return new(Bool(true), Float64(0), Bool(false),
                                            rand(RandomDeviceInstance, [(1, 0), (0, 1), (-1, 0), (0, -1)]),
                                            Array{Any, 1}());
    end

    function Agent{T <: AgentProgram}(ap::T)
        new_agent = new(Bool(true), Float64(0), Bool(false),
                                                rand(RandomDeviceInstance, [(1, 0), (0, 1), (-1, 0), (0, -1)]),
                                                Array{Any, 1}());   #program is undefined
        new_agent.program = ap;
        return new_agent;
    end
end

type Wumpus <: EnvironmentAgent
    alive::Bool
    performance::Float64
    bump::Bool
    heading::Tuple{Any, Any}
    holding::Array{Any, 1}
    program::AgentProgram
    location::Tuple{Any, Any}       #initialized when adding agent to environment

    function Wumpus()
        return new(Bool(true), Float64(0), Bool(false),
                                            rand(RandomDeviceInstance, [(1, 0), (0, 1), (-1, 0), (0, -1)]),
                                            Array{Any, 1}());
    end

    function Wumpus{T <: AgentProgram}(ap::T)
        new_agent = new(Bool(true), Float64(0), Bool(false),
                                                rand(RandomDeviceInstance, [(1, 0), (0, 1), (-1, 0), (0, -1)]),
                                                Array{Any, 1}());   #program is undefined
        new_agent.program = ap;
        return new_agent;
    end
end

type Explorer <: EnvironmentAgent
    alive::Bool
    performance::Float64
    bump::Bool
    heading::Tuple{Any, Any}
    holding::Array{Any, 1}
    program::AgentProgram
    location::Tuple{Any, Any}       #initialized when adding agent to environment

    function Explorer()
        return new(Bool(true), Float64(0), Bool(false),
                                            rand(RandomDeviceInstance, [(1, 0), (0, 1), (-1, 0), (0, -1)]),
                                            Array{Any, 1}());
    end

    function Explorer{T <: AgentProgram}(ap::T)
        new_agent = new(Bool(true), Float64(0), Bool(false),
                                                rand(RandomDeviceInstance, [(1, 0), (0, 1), (-1, 0), (0, -1)]),
                                                Array{Any, 1}());   #program is undefined
        new_agent.program = ap;
        return new_agent;
    end
end

function isAlive{T <: Agent}(a::T)
    return a.alive;
end

function setAlive{T <: Agent}(a::T, bv::Bool)
    a.alive = bv;
    nothing;
end

#=

    Obstacle Environment Objects

=#

abstract Obstacle <: EnvironmentObject

type Wall <: Obstacle
    location::Tuple{Any, Any}

    function Wall()
        return new();
    end
end

#=

    Vacuum Environment Objects

=#

type Dirt <: EnvironmentObject
    location::Tuple{Any, Any}

    function Dirt()
        return new();
    end
end

#=

    Wumpus Environment Objects

=#

type Gold <: EnvironmentObject
    location::Tuple{Any, Any}

    function Gold()
        return new();
    end
end

type Pit <: EnvironmentObject
    location::Tuple{Any, Any}

    function Pit()
        return new();
    end
end

type Arrow <: EnvironmentObject
    location::Tuple{Any, Any}

    function Array()
        return new();
    end
end

#=

    Implement execute() methods for implemented AgentPrograms.

=#

loc_A = (0, 0)
loc_B = (1, 0)

function execute(ap::TableDrivenAgentProgram, percept::Tuple{Any, Any})
    push!(ap.percepts, percept);
    local action;
    if (haskey(ap.table, Tuple((ap.percepts...))))
        action = ap.table[Tuple((ap.percepts...))]  #convert percept sequence to tuple
        if (ap.isTracing)
            @printf("%s perceives %s and does %s\n", string(typeof(ap)), string(percept), action.name);
        end
    else
        #The table is not complete with all possible percept sequences.
        #So, this program should now behave like a ReflexVacuumAgentProgram.
        if (percept[1] == loc_A)
            action = "Right";
        elseif (percept[1] == loc_B)
            action = "Left";
        end
    end
    return action;
end

function execute(ap::ReflexVacuumAgentProgram, location_status::Tuple{Any, Any})
    local location = location_status[1];
    local status = location_status[2];
    if (status == "Dirty")
        if (ap.isTracing)
            @printf("%s perceives %s and does %s\n", string(typeof(ap)), string(location_status), "Suck");
        end
        return "Suck";
    elseif (location == loc_A)
        if (ap.isTracing)
            @printf("%s perceives %s and does %s\n", string(typeof(ap)), string(location_status), "Right");
        end
        return "Right";
    elseif (location == loc_B)
        if (ap.isTracing)
            @printf("%s perceives %s and does %s\n", string(typeof(ap)), string(location_status), "Left");
        end
        return "Left";
    end
end

function execute(ap::ModelBasedVacuumAgentProgram, location_status::Tuple{Any, Any})
    local location = location_status[1];
    local status = location_status[2];
    ap.model[location] = status;                            #update existing model
    if (ap.model[loc_A] == ap.model[loc_B] == "Clean")      #return "NoOp" when no work is necessary
        if (ap.isTracing)
            @printf("%s perceives %s and does %s\n", string(typeof(ap)), string(location_status), "NoOp");
        end
        return "NoOp";
    elseif (status == "Dirty")
        if (ap.isTracing)
            @printf("%s perceives %s and does %s\n", string(typeof(ap)), string(location_status), "Suck");
        end
        return "Suck";
    elseif (location == loc_A)
        if (ap.isTracing)
            @printf("%s perceives %s and does %s\n", string(typeof(ap)), string(location_status), "Right");
        end
        return "Right";
    elseif (location == loc_B)
        if (ap.isTracing)
            @printf("%s perceives %s and does %s\n", string(typeof(ap)), string(location_status), "Left");
        end
        return "Left";
    end
end

function execute(ap::RandomAgentProgram, percept::Tuple{Any, Any})
    return rand(RandomDeviceInstance, ap.actions);
end

function rule_match(state::String, rules::Array{Rule, 1})
    for element in rules
        if (state == element.condition)
            return element;
        end
    end
    return C_NULL;                  #the function did not find a matching rule
end

function interpret_input{T <: AgentProgram}(ap::T, percept::Tuple{Any, Any})        #implement this later
    println("interpret_input() is not implemented yet for ", typeof(ap), "!");
    nothing;
end

function update_state{T <: AgentProgram}(ap::T, percept::Tuple{Any, Any})           #implement this later
    println("update_state() is not implemented yet for ", typeof(ap), "!");
    nothing;
end

function execute(ap::SimpleReflexAgentProgram, percept::Tuple{Any, Any})
    #the agent acts according to the given percept (Fig. 2.10)
    local state = interpret_input(ap, percept);     #generate condition string from given percept
    local rule = rule_match(state, ap.rules);
    local action = rule.action;
    if (ap.isTracing)
        @printf("%s perceives %s and does %s\n", string(typeof(ap)), string(percept), action);
    end
    return action;
end

function execute(ap::ModelBasedReflexAgentProgram, percept::Tuple{Any, Any})
    #the agent acts according to the agent state and model and given percept (Fig. 2.12)

    #ap.state <- update-state(ap.state, ap.action, percept, ap.model);  #set new state
    ap.state = update_state(ap, percept);
    local rule = rule_match(ap.state, ap.rules);
    ap.action = rule.action;
    if (ap.isTracing)
        @printf("%s perceives %s and does %s\n", string(typeof(ap)), string(percept), ap.action);
    end
    return ap.action;
end

#=

    Load a implemented AgentProgram into a Agent.

=#

function TableDrivenVacuumAgent()
    #dictionary representation of table (Fig. 2.3) of percept sequences (key) mappings to actions (value).
    local table = Dict{Any, Any}([
                Pair(((loc_A, "Clean"),), "Right"),
                Pair(((loc_A, "Dirty"),), "Suck"),
                Pair(((loc_B, "Clean"),), "Left"),
                Pair(((loc_B, "Dirty"),), "Suck"),
                Pair(((loc_A, "Clean"), (loc_A, "Clean")), "Right"),
                Pair(((loc_A, "Clean"), (loc_A, "Dirty")), "Suck"),
                # ...
                Pair(((loc_A, "Clean"), (loc_A, "Clean"), (loc_A, "Clean")), "Right"),
                Pair(((loc_A, "Clean"), (loc_A, "Clean"), (loc_A, "Dirty")), "Suck"),
                # ...
                ]);
    return Agent(TableDrivenAgentProgram(table_dict=table));
end

function ReflexVacuumAgent()
    #Return a reflex agent for the two-state vacuum environment (Fig. 2.8).
    return Agent(ReflexVacuumAgentProgram());
end

function ModelBasedVacuumAgent()
    #Return a agent that tracks statuses of clean and dirty locations.
    return Agent(ModelBasedVacuumAgentProgram(model=Dict{Any, Any}([
            Pair(loc_A, Void),
            Pair(loc_B, Void),
            ])));
end

function RandomVacuumAgent()
    return Agent(RandomAgentProgram(["Right", "Left", "Suck", "NoOp"]));
end

abstract Environment;               #declare Environment as a supertype for Environment implementations

abstract TwoDimensionalEnvironment <: Environment;

#XYEnvironment is a 2-dimensional Environment implementation with obstacles.
#Agents perceive their location as a tuple of objects within perceptible_distance radius.
#This environment does not update agent performance measures.
type XYEnvironment <: TwoDimensionalEnvironment
    objects::Array{EnvironmentObject, 1}
    agents::Array{Agent, 1}                 #agents found in this field should also be found in the objects field
    width::Float64
    height::Float64
    perceptible_distance::Float64

    function XYEnvironment()
        local xy = new(Array{EnvironmentObject, 1}(), Array{Agent, 1}(), Float64(10), Float64(10), Float64(1));
        add_walls(xy);
        return xy;
    end
end

#VacuumEnvironment is a 2-dimensional Environment implementation with obstacles.
#Agents can perceive their location as "Dirty" or "Clean".
#Agent performance measures are updated when Dirt is removed or a non-NoOp action is executed.
type VacuumEnvironment <: TwoDimensionalEnvironment
    objects::Array{EnvironmentObject, 1}
    agents::Array{Agent, 1}                 #agents found in this field should also be found in the objects field
    width::Float64
    height::Float64
    perceptible_distance::Float64

    function VacuumEnvironment()
        local ve = new(Array{EnvironmentObject, 1}(), Array{Agent, 1}(), Float64(10), Float64(10), Float64(1));
        add_walls(ve);
        return ve;
    end
end

#TrivialVacuumEnvironment has 2 possible locations: loc_A and loc_B.
#The status of those locations can be either "Dirty" or "Clean".
type TrivialVacuumEnvironment <: Environment
    objects::Array{EnvironmentObject, 1}
    agents::Array{Agent, 1}
    status::Dict{Tuple{Any, Any}, String}
    perceptible_distance::Float64

    function TrivialVacuumEnvironment()
        local tve = new(
                Array{EnvironmentObject, 1}(),
                Array{Agent, 1}(),
                Dict{Tuple{Any, Any}, String}([
                    Pair(loc_A, rand(RandomDeviceInstance, ["Clean", "Dirty"])),
                    Pair(loc_B, rand(RandomDeviceInstance, ["Clean", "Dirty"])),
                    ]), Float64(1));
        return tve;
    end
end

type WumpusEnvironment <: TwoDimensionalEnvironment
    objects::Array{EnvironmentObject, 1}
    agents::Array{Array, 1}
    width::Float64
    height::Float64
    perceptible_distance::Float64

    function WumpusEnvironment()
        local we = new(Array{EnvironmentObject, 1}(), Array{Agent, 1}(), Float64(10), Float64(10), Float64(1));
        add_walls(we);
        return we;
    end
end

"""
    percept(e, agent, act)

Returns a percept representing what the agent perceives in the enviroment.
"""
function percept{T1 <: Environment, T2 <: EnvironmentAgent, T3 <: String}(e::T1, a::T2, act::T3)    #implement this later
    println("percept() is not implemented yet for ", typeof(e), "!");
    nothing;
end

function percept(e::VacuumEnvironment, a::Agent)
    local status = if_(some_objects_at(a.location, Dirt), "Dirty", "Clean");
    local bump = if_(a.bump, "Bump", "None");
    return (status, bump)
end

"""
    objects_near(e, location)
    objects_near(e, location, radius)

Return a list of EnvironmentObjects within the radius of a given location.
"""
function objects_near(e::XYEnvironment, loc::Tuple{Any, Any}; radius::Union{Void, Float64}=nothing)
    if (typeof(radius) <: Void)
        radius = e.perceptible_distance;
    end
    sq_radius = radius * radius;
    return [obj for obj in e.objects if (utils.distance2(loc, obj.location) <= sq_radius)];
end

function percept(e::XYEnvironment, a::Agent)
    #this percept might not consist of exactly 2 elements
    return Tuple(([string(typeof(obj)) for obj in objects_near(a.location)]...));
end

function percept(e::TrivialVacuumEnvironment, a::Agent)
    return (a.location, e.status[a.location]);
end

function get_objects_at{T <: Environment}(e::T, loc::Tuple{Any, Any}, objType::DataType)
    if (objType <: EnvironmentObject)
        return [obj for obj in e.objects if (typeof(obj) <: objType && obj.location == loc)];
    else
        error(@sprintf("InvalidEnvironmentObjectError: %s is not a subtype of EnvironmentObject!", string(typeof(objType))));
    end
end

function some_objects_at{T <: Environment}(e::T, loc::Tuple{Any, Any}, objType::DataType)
    object_array = get_objects_at(e, loc, objType);
    if (length(object_array) == 0)
        return false;
    else
        return true;
    end
end

function is_done{T <:Environment}(e::T)
    for a in e.agents
        if (a.alive)
            return false;
        end
    end
    return true;
end

function step{T <: Environment}(e::T)
    if (!is_done(e))
        local actions = [execute(agent.program, percept(e, agent)) for agent in e.agents];
        for t in zip(e.agents, actions)
            local agent = t[1];
            local action = t[2];
            execute_action(e, agent, action);
        end
        exogenous_change(e);
    end
end

function run{T <: Environment}(e::T; steps::Int64=1000)
    for i in range(0, steps)
        if (is_done(e))
            break;
        end
        step(e);
    end
end

function exogenous_change{T <: Environment}(e::T)   #implement this later
    #comment the following line to reduce verbosity
    #println("exogenous_change() not yet implemented for ", typeof(e), "!");
    nothing;
end

function default_location{T1 <: Environment, T2 <: EnvironmentObject}(e::T1, obj::T2)   #implement this later
    return false;
end

function default_location{T <: EnvironmentObject}(e::TrivialVacuumEnvironment, obj::T)
    return rand(RandomDeviceInstance, [loc_A, loc_B]);
end

function default_location{T <: EnvironmentObject}(e::XYEnvironment, obj::T)
    return (rand(RandomDeviceInstance, range(0, e.width)), rand(RandomDeviceInstance, range(0, e.height)));
end

function environment_objects{T <: Environment}(e::T)
    return [];
end

function environment_objects(e::VacuumEnvironment)
    #ReflexVacuumAgent, RandomVacuumAgent, TableDrivenVacuumAgent, and ModelBasedVacuumAgent
    #are functions that generate new agents with their respective AgentPrograms
    return [Wall, Dirt, ReflexVacuumAgent, RandomVacuumAgent, TableDrivenVacuumAgent, ModelBasedVacuumAgent];
end

function environment_objects(e::TrivialVacuumEnvironment)
    #ReflexVacuumAgent, RandomVacuumAgent, TableDrivenVacuumAgent, and ModelBasedVacuumAgent
    #are functions that generate new agents with their respective AgentPrograms
    return [Wall, Dirt, ReflexVacuumAgent, RandomVacuumAgent, TableDrivenVacuumAgent, ModelBasedVacuumAgent];
end

function environment_objects(e::WumpusEnvironment)
    return [Wall, Gold, Pit, Arrow, Wumpus, Explorer];
end

function execute_action{T1 <: Environment, T2 <: EnvironmentAgent}(e::T1, a::T2, act::String)   #implement this later
    println("execute_action() is not implemented yet for ", string(typeof(e)), "!");
    nothing;
end

function execute_action(e::XYEnvironment, a::EnvironmentAgent, act::String)
    a.bump = false;
    if (act == "TurnRight")
        a.heading = utils.turn_heading(a.heading, -1);
    elseif (act == "TurnLeft")
        a.heading = utils.turn_heading(a.heading, 1);
    elseif (act == "Foward")
        move_to(e, a, utils.vector_add_tuples(a.heading, a.location));
    elseif (act == "Release")
        if (length(a.holding) > 0)
            pop!(a.holding);
        end
    end
    nothing;
end

function execute_action(e::VacuumEnvironment, a::EnvironmentAgent, act::String)
    if (act == "Suck")
        local dirt_array = get_objects_at(e, a.location, Dirt);
        if (length(dirt_array) > 0)
            local dirt = pop!(dirt);
            delete_object(e, dirt);
            a.performance = a.performance + 100;
        end
    else
        a.bump = false;
        if (act == "TurnRight")
            a.heading = utils.turn_heading(a.heading, -1);
        elseif (act == "TurnLeft")
            a.heading = utils.turn_heading(a.heading, 1);
        elseif (act == "Foward")
            move_to(e, a, utils.vector_add_tuples(a.heading, a.location));
        elseif (act == "Release")
            if (length(a.holding) > 0)
                pop!(a.holding);
            end
        end
    end
    if (act != "NoOp")
        a.performance = a.performance - 1;
    end
    nothing;
end

function execute_action(e::TrivialVacuumEnvironment, a::EnvironmentAgent, act::String)
    if (act == "Right")
        a.location = loc_B;
        a.performance = a.performance - 1;
    elseif (act == "Left")
        a.location = loc_A;
        a.performance = a.performance - 1;
    elseif (act == "Suck")
        if (e.status[a.location] == "Dirty")
            a.performance = a.performance + 10;
        end
        e.status[a.location] = "Clean";
    end
    nothing;
end

function add_object{T1 <: Environment, T2 <: EnvironmentObject}(e::T1, obj::T2; location::Union{Void, Tuple{Any, Any}}=nothing)
    if (!(obj in e.objects))
        if (!(typeof(location) <: Void))
            obj.location = location;
        else
            obj.location = default_location(e, obj);
        end
        push!(e.objects, obj);
        if (typeof(obj) <: EnvironmentAgent)
            obj.performance = Float64(0);
            push!(e.agents, obj);
        end
    else
        println("add_object(): object already exists in environment!");
    end
    nothing;
end

function delete_object{T1 <: Environment, T2 <: EnvironmentObject}(e::T1, obj::T2)
    local i = utils.index(e.objects, obj);
    if (i > -1)
        deleteat!(e.objects, i);
    end
    i = utils.index(e.agents, obj);
    if (i > -1)
        deleteat!(e.agents, i);
    end
    nothing;
end

function add_walls{T <: TwoDimensionalEnvironment}(e::T)
    for x in range(0, e.width)
        add_object(Wall(), location=(x, 0));
        add_object(Wall(), location=(x, e.height - 1));
    end
    for y in range(0, e.height)
        add_object(Wall(), location=(0, y));
        add_object(Wall(), location=(e.width - 1, 0));
    end
    nothing;
end

"""
    move_to(e, obj, dest)

Move the EnvironmentObject to the destination given.
"""
function move_to{T <: TwoDimensionalEnvironment}(e::T, obj::EnvironmentObject, destination::Tuple{Any, Any})
    obj.bump = some_objects_at(e, destination, Wall);   #Wall is a subtype of Obstacle, not an alias
    if (!obj.bump)
        obj.location = destination;
    end
    nothing;
end

function run_once(e::Environment, AgentGen::Function, step_count::Int)
    local agent = AgentGen();
    add_object(e, agent);
    run(e, steps=step_count);
    return agent.performance;
end

"""
    test_agent(agentgeneratorfunction, steps, envs)

Calculates the average of the scores of running the given Agent in each of the environments.
"""
function test_agent{T <: Environment}(AgentGenerator::Function, steps::Int, envs::Array{T, 1})
    return mean([run_once(envs[i], AgentGenerator, steps) for i in 1:length(envs)]);
end

"""
    compare_agents()

Creates an array of 'n' Environments and runs each Agent in each separate copy of the Environment
array for 'steps' times. Then, return a list of (agent, average_score) tuples.
"""
function compare_agents(EnvironmentGenerator::DataType, AgentGenerators::Array{Function, 1}; n::Int64=10, steps::Int64=1000)
    local envs = [EnvironmentGenerator() for i in range(0, n)];
    return [(string(typeof(A)), test_agent(A, steps, deepcopy(envs))) for A in AgentGenerators];
end
