include("utils.jl");

typealias Percept Tuple{Any, Any}

typealias Action String;

abstract AgentProgram;      #declare AgentProgram as a supertype for AgentProgram implementations

#=

    Define a global execute() function to be implemented for each respective
    AgentProgram DataType implementation.
        
=#

function execute{T <: AgentProgram}(ap::T, p::Percept)      #implement functionality later
    #comment the following line to reduce verbosity
    #println("execute() is not implemented yet for ", typeof(ap), "!");
    nothing;
end

type TableDrivenAgentProgram <: AgentProgram
    isTracing::Bool
    percepts::Array{Percept, 1}
    table::Dict{Any, Any}

    function TableDrivenAgentProgram(;table_dict=C_NULL, trace=false)
        if (table_dict == C_NULL)           #no table given, create empty dictionary
            return new(Bool(trace), Array{Percept, 1}(), Dict{Any, Any}());
        else
            return new(Bool(trace), Array{Percept, 1}(), table_dict);
        end
    end
end

type ReflexVacuumAgentProgram <: AgentProgram
    isTracing::Bool

    function ReflexVacuumAgentProgram(;trace=false)
        return new(Bool(trace));
    end
end

type ModelBasedVacuumAgentProgram <: AgentProgram
    isTracing::Bool
    model::Dict{Any, Any}

    function ModelBasedVacuumAgentProgram(;trace=false, model=Void)
        if (typeof(model) <: Dict{Any, Any})
            new_ap = new(Bool(trace));
            new_ap.model = deepcopy(model);
            return new_ap;
        else
            return new(Bool(trace), Dict{Any, Any}());
        end
    end
end

type Rule   #condition-action rules
    condition::String
    action::Action

    function Rule(cond::String, act::Action)
        return new(cond, act);
    end
end

type SimpleReflexAgentProgram <: AgentProgram
    isTracing::Bool
    rules::Array{Rule, 1}

    function SimpleReflexAgentProgram(rules_array::Array{Rule, 1};trace=false)
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
    action::Action          #most recent action, initialized to empty string ""

    function ModelBasedReflexAgentProgram(state, model, rules;trace=false)
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
    program::AgentProgram
    location::Tuple{Any, Any}       #initialized when adding agent to environment

    function Agent()
        return new(Bool(true), Float64(0), Bool(false));
    end

    function Agent{T <: AgentProgram}(ap::T)
        new_agent = new(Bool(true), Float64(0), Bool(false));   #program is undefined
        new_agent.program = ap;
        return new_agent;
    end
end

type Wumpus <: EnvironmentAgent
    alive::Bool
    performance::Float64
    bump::Bool
    program::AgentProgram
    location::Tuple{Any, Any}       #initialized when adding agent to environment

    function Wumpus()
        return new(Bool(true), Float64(0), Bool(false));
    end

    function Wumpus{T <: AgentProgram}(ap::T)
        new_agent = new(Bool(true), Float64(0), Bool(false));   #program is undefined
        new_agent.program = ap;
        return new_agent;
    end
end

type Explorer <: EnvironmentAgent
    alive::Bool
    performance::Float64
    bump::Bool
    program::AgentProgram
    location::Tuple{Any, Any}       #initialized when adding agent to environment

    function Explorer()
        return new(Bool(true), Float64(0), Bool(false));
    end

    function Explorer{T <: AgentProgram}(ap::T)
        new_agent = new(Bool(true), Float64(0), Bool(false));   #program is undefined
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

function execute(ap::TableDrivenAgentProgram, percept::Percept)
    ap.percepts.push(percept);
    action = ap.table[Tuple((ap.percepts...))]  #convert percept sequence to tuple
    if (ap.isTracing)
        @printf("%s perceives %s and does %s\n", string(typeof(ap)), string(percept), action.name);
    end
    return action;
end

function execute(ap::ReflexVacuumAgentProgram, location_status::Percept)
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

function execute(ap::ModelBasedVacuumAgentProgram, location_status::Percept)
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

function rule_match(state::String, rules::Array{Rule, 1})
    for element in rules
        if (state == element.condition)
            return element;
        end
    end
    return C_NULL;                  #the function did not find a matching rule
end

function interpret_input{T <: AgentProgram}(ap::T, percept::Percept)        #implement this later
    println("interpret_input() is not implemented yet for ", typeof(ap), "!");
    nothing;
end

function update_state{T <: AgentProgram}(ap::T, percept::Percept)           #implement this later
    println("update_state() is not implemented yet for ", typeof(ap), "!");
    nothing;
end

function execute(ap::SimpleReflexAgentProgram, percept::Percept)
    #the agent acts according to the given percept (Fig. 2.10)
    local state = interpret_input(ap, percept);     #generate condition string from given percept
    local rule = rule_match(state, ap.rules);
    local action = rule.action;
    if (ap.isTracing)
        @printf("%s perceives %s and does %s\n", string(typeof(ap)), string(percept), action);
    end
    return action;
end

function execute(ap::ModelBasedReflexAgentProgram, percept::Percept)
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

abstract Environment;               #declare Environment as a supertype for Environment implementations

abstract TwoDimensionalEnvironment <: Environment;

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

type TrivialVacuumEnvironment <: TwoDimensionalEnvironment
    objects::Array{EnvironmentObject, 1}
    agents::Array{Agent, 1}
    status::Dict{Tuple{Any, Any}, String}
    perceptible_distance::Float64

    function TrivialVacuumEnvironment()
        local tve = new(
                Array{EnvironmentObject, 1}(),
                Array{Agent, 1}(),
                Dict{Tuple{Any, Any}, String}([
                    Pair(loc_A, rand(RandomDevice(), ["Clean", "Dirty"])),
                    Pair(loc_B, rand(RandomDevice(), ["Clean", "Dirty"])),
                    ]), Float64(1));
        add_walls(tve);
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

function percept{T1 <: Environment, T2 <: EnvironmentAgent, T3 <: Action}(e::T1, a::T2, act::T3)        #implement this later
    println("percept() is not implemented yet for ", typeof(e), "!");
    nothing;
end

function get_objects_at{T <: Environment}(e::T, loc::Tuple{Any, Any}, objType::DataType)
    if (objType <: EnvironmentObject)
        return [obj for obj in e.objects if (typeof(obj) == objType && obj.location == loc)];
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

function exogenous_change{T <: Environment}(e::T)   #implement this later
    #comment the following line to reduce verbosity
    #println("exogenous_change() not yet implemented for ", typeof(e), "!");
    nothing;
end

function default_location{T1 <: Environment, T2 <: EnvironmentObject}(e::T1, obj::T2)   #implement this later
    return false;
end

function default_location{T <: EnvironmentObject}(e::TrivialVacuumEnvironment, obj::T)
    return rand(RandomDevice(), [loc_A, loc_B]);
end

function default_location{T <: EnvironmentObject}(e::XYEnvironment, obj::T)
    return (rand(RandomDevice(), range(0, e.width)), rand(RandomDevice(), range(0, e.height)));
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

function percept(e::VacuumEnvironment, a::Agent)
    local status = if_(some_objects_at(a.location, Dirt), "Dirty", "Clean");
    local bump = if_(a.bump, "Bump", "None");
    return (status, bump)
end

function objects_near(e::XYEnvironment, loc::Tuple{Any, Any}; radius=C_NULL)
    if (radius == C_NULL)
        radius = e.perceptible_distance;
    end
    sq_radius = radius * radius;
    return [obj for obj in e.objects if (utils.distance2(loc, obj.location) <= sq_radius)];
end

function percept(e::XYEnvironment, a::Agent)
    return [string(typeof(obj)) for obj in objects_near(a.location)];
end

function execute_action{T1 <: Environment, T2 <: EnvironmentAgent}(e::T1, a::T2, act::Action)   #implement this later
    println("execute_action() is not implemented yet for ", string(typeof(e)), "!");
    nothing;
end

function add_object{T1 <: Environment, T2 <: EnvironmentObject}(e::T1, obj::T2; location=C_NULL)
    if (!(obj in e.objects))
        if (location != C_NULL)
            obj.location = location;
        else
            obj.location = default_location(e, obj);
        end
        append!(e.objects, obj);
        if (typeof(obj) <: EnvironmentAgent)
            obj.performance = Float64(0);
            append!(e.agents, obj);
        end
    else
        println("add_object(): object already exists in environment!");
    end
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
end

function move_to{T <: TwoDimensionalEnvironment}(e::T, obj::EnvironmentObject, destination::Tuple{Any, Any})
    obj.bump = some_objects_at(e, destination, Wall);   #Wall is a subtype of Obstacle, not an alias
    if (!obj.bump)
        obj.location = destination;
    end
end
