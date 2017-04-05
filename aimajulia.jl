module aimajulia;

include("utils.jl");

using aimajulia.utils;

export Action, Percept;

typealias Action String;

typealias Percept Tuple{Any, Any};

include("agents.jl");

include("search.jl");

include("games.jl");

include("csp.jl");

end;