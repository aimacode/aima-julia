include("utils.jl");

typealias Percept Tuple{Any, Any}

typealias Action String;

abstract AgentProgram;		#declare AgentProgram as a supertype for AgentProgram implementations

#=

	Define a global execute() function to be implemented for each respective
	AgentProgram DataType implementation.
		
=#

function execute{T <: AgentProgram}(ap::T, p::Percept)		#implement functionality later
	println("execute() not yet implemented for ", typeof(ap), "!");
	nothing;
end

type TableDrivenAgentProgram <: AgentProgram
	isTracing::Bool
	percepts::Array{Percept, 1}
	table::Dict{Any, Any}

	function TableDrivenAgentProgram(;table_dict=C_NULL, trace=false)
		if (table_dict == C_NULL)			#no table given, create empty dictionary
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

type Rule	#condition-action rules
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
	action::Action			#most recent action, initialized to empty string ""

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

abstract EnvironmentObject;			#declare EnvironmentObject as a supertype for EnvironmentObject implementations

#the EnvironmentAgent implementations exist in the environment like other EnvironmentObjects such as Gold or Dirt
abstract EnvironmentAgent <: EnvironmentObject;

type Agent <: EnvironmentAgent
	alive::Bool
	performance::Float64
	bump::Bool
	program::AgentProgram
	location::Tuple{Any, Any}		#initialized when adding agent to environment

	function Agent()
		return new(Bool(true), Float64(0), Bool(false));
	end

	function Agent{T <: AgentProgram}(ap::T)
		new_agent = new(Bool(true), Float64(0), Bool(false));	#program is undefined
		new_agent.program = ap;
		return new_agent;
	end
end

type Wumpus <: EnvironmentAgent
	alive::Bool
	performance::Float64
	bump::Bool
	program::AgentProgram
	location::Tuple{Any, Any}		#initialized when adding agent to environment

	function Wumpus()
		return new(Bool(true), Float64(0), Bool(false));
	end

	function Wumpus{T <: AgentProgram}(ap::T)
		new_agent = new(Bool(true), Float64(0), Bool(false));	#program is undefined
		new_agent.program = ap;
		return new_agent;
	end
end

type Explorer <: EnvironmentAgent
	alive::Bool
	performance::Float64
	bump::Bool
	program::AgentProgram
	location::Tuple{Any, Any}		#initialized when adding agent to environment

	function Explorer()
		return new(Bool(true), Float64(0), Bool(false));
	end

	function Explorer{T <: AgentProgram}(ap::T)
		new_agent = new(Bool(true), Float64(0), Bool(false));	#program is undefined
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
	action = ap.table[Tuple((ap.percepts...))]	#convert percept sequence to tuple
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
	ap.model[location] = status;							#update existing model
	if (ap.model[loc_A] == ap.model[loc_B] == "Clean")		#return "NoOp" when no work is necessary
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
	return C_NULL;					#the function did not find a matching rule
end

function interpret_input{T <: AgentProgram}(ap::T, percept::Percept)		#implement this later
	println("interpret_input() is not yet implemented for ", typeof(ap), "!");
	nothing;
end

function update_state{T <: AgentProgram}(ap::T, percept::Percept)			#implement this later
	println("update_state() is not yet implemented for ", typeof(ap), "!");
	nothing;
end

function execute(ap::SimpleReflexAgentProgram, percept::Percept)
	#the agent acts according to the given percept (Fig. 2.10)
	local state = interpret_input(ap, percept);		#generate condition string from given percept
	local rule = rule_match(state, ap.rules);
	local action = rule.action;
	if (ap.isTracing)
		@printf("%s perceives %s and does %s\n", string(typeof(ap)), string(percept), action);
	end
	return action;
end

function execute(ap::ModelBasedReflexAgentProgram, percept::Percept)
	#the agent acts according to the agent state and model and given percept (Fig. 2.12)

	#ap.state <- update-state(ap.state, ap.action, percept, ap.model);	#set new state
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

abstract Environment;				#declare Environment as a supertype for Environment implementations

type XYEnvironment <: Environment
	objects::Array{EnvironmentObject, 1}
	agents::Array{Agent, 1}					#agents found in this field should also be found in the objects field
	width::Float64
	height::Float64

	function XYEnvironment()
		return new(Array{EnvironmentObject, 1}(), Array{Agent, 1}(), Float64(10), Float64(10));
	end
end

type VacuumEnvironment <: Environment
	objects::Array{EnvironmentObject, 1}
	agents::Array{Agent, 1}					#agents found in this field should also be found in the objects field
	width::Float64
	height::Float64

	function VacuumEnvironment()
		return new(Array{EnvironmentObject, 1}(), Array{Agent, 1}(), Float64(10), Float64(10));
	end
end

type TrivialVacuumEnvironment <: Environment
	objects::Array{EnvironmentObject, 1}
	agents::Array{Agent, 1}
	status::Dict{Tuple{Any, Any}, String}

	function TrivialVacuumEnvironment()
		return new(
				Array{EnvironmentObject, 1}(),
				Array{Agent, 1}(),
				Dict{Tuple{Any, Any}, String}([
					Pair(loc_A, rand(RandomDevice(), ["Clean", "Dirty"])),
					Pair(loc_B, rand(RandomDevice(), ["Clean", "Dirty"])),
					]));
	end
end

type WumpusEnvironment <: Environment
	objects::Array{EnvironmentObject, 1}
	agents::Array{Array, 1}
	width::Float64
	height::Float64

	function WumpusEnvironment()
		return new(Array{EnvironmentObject, 1}(), Array{Agent, 1}(), Float64(10), Float64(10));
	end
end

function percept{T1 <: Environment, T2 <: EnvironmentAgent, T3 <: Action}(e::T1, a::T2, act::T3)		#implement this later
	println("percept() not yet implemented for ", typeof(e), "!");
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

function exogenous_change{T <: Environment}(e::T)	#implement this later
	#println("exogenous_change() not yet implemented for ", typeof(e), "!");#comment this line to reduce verbosity
	nothing;
end

function default_location{T1 <: Environment, T2 <: EnvironmentObject}(e::T1, obj::T2)	#implement this later
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
	return [Wall, Dirt, ReflexVacuumAgent, RandomVacuumAgent, TableDrivenVacuumAgent, ModelBasedVacuumAgent];
end

function environment_objects(e::TrivialVacuumEnvironment)
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
