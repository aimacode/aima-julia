# aima-julia

[![Build Status](https://travis-ci.org/aimacode/aima-julia.svg?branch=master)](https://travis-ci.org/aimacode/aima-julia)

Julia (v0.5+) implementation of the algorithms found in "Artificial Intelligence: A Modern Approach".

Using aima-julia for portable purposes
--------------------------------------

Include the following lines in all files within the same directory.

~~~
include("aimajulia.jl");
using aimajulia;
~~~

Index of Algorithms
-------------------

| **Figure** | **Name (in 3<sup>rd</sup> edition)** | **Name (in repository)** | **File**
|:--------|:-------------------|:---------|:-----------|
| 2.1     | Environment        | `Environment` | [`src/agents.jl`][agents] |
| 2.1     | Agent              | `Agent` | [`src/agents.jl`][agents] |
| 2.3     | Table-Driven-Vacuum-Agent | `TableDrivenVacuumAgent` | [`src/agents.jl`][agents] |
| 2.7     | Table-Driven-Agent | `TableDrivenAgentProgram` | [`src/agents.jl`][agents] |
| 2.8     | Reflex-Vacuum-Agent | `ReflexVacuumAgent` | [`src/agents.jl`][agents] |
| 2.10    | Simple-Reflex-Agent | `SimpleReflexAgent` | [`src/agents.jl`][agents] |
| 2.12    | Model-Based-Reflex-Agent | `ModelBasedReflexAgentProgram` | [`src/agents.jl`][agents] |
| 3       | Problem            | `Problem` | [`src/search.jl`][search] |
| 3       | Node               | `Node` | [`src/search.jl`][search] |
| 3       | Queue              | `Queue` | [`src/utils.jl`][utils] |
| 3.1     | Simple-Problem-Solving-Agent | `SimpleProblemSolvingAgent` | [`src/search.jl`][search] |
| 3.2     | Romania            | `romania` | [`src/search.jl`][search] |
| 3.7     | Tree-Search        | `tree_search` | [`src/search.jl`][search] |
| 3.7     | Graph-Search        | `graph_search` | [`src/search.jl`][search] |
| 3.11    | Breadth-First-Search        | `breadth_first_search` | [`src/search.jl`][search] |
| 3.14    | Uniform-Cost-Search        | `uniform_cost_search` | [`src/search.jl`][search] |
| 3.17    | Depth-Limited-Search | `depth_limited_search` | [`src/search.jl`][search] |
| 3.18    | Iterative-Deepening-Search | `iterative_deepening_search` | [`src/search.jl`][search] |
| 3.22    | Best-First-Search  | `best_first_graph_search` | [`src/search.jl`][search] |
| 3.24    | A\*-Search        | `astar_search` | [`src/search.jl`][search] |
| 3.26    | Recursive-Best-First-Search | `recursive_best_first_search` | [`src/search.jl`][search] |
| 4.2     | Hill-Climbing      | `hill_climbing` | [`src/search.jl`][search] |
| 4.5     | Simulated-Annealing | `simulated_annealing` | [`src/search.jl`][search] |
| 4.8     | Genetic-Algorithm  | `genetic_algorithm` | [`src/search.jl`][search] |
| 4.11    | And-Or-Graph-Search | `and_or_graph_search` | [`src/search.jl`][search] |
| 4.21    | Online-DFS-Agent   |  |  |
| 4.24    | LRTA\*-Agent       |  |  |
| 5.3     | Minimax-Decision   | `minimax_decision` | [`src/games.jl`][games] |
| 5.7     | Alpha-Beta-Search  | `alphabeta_search` | [`src/games.jl`][games] |
| 6       | CSP                | `CSP` | [`src/csp.jl`][csp] |
| 6.3     | AC-3               | `AC3` | [`src/csp.jl`][csp] |
| 6.5     | Backtracking-Search | `backtracking_search` | [`src/csp.jl`][csp] |
| 6.8     | Min-Conflicts      | `min_conflicts` | [`src/csp.jl`][csp] |
| 6.11    | Tree-CSP-Solver    | `tree_csp_solver` | [`src/csp.jl`][csp] |
| 7       | KB                 | `KnowledgeBase` | [`src/logic.jl`][logic] |
| 7.1     | KB-Agent           | `KnowledgeBaseAgentProgram` | [`src/logic.jl`][logic] |
| 7.7     | Propositional Logic Sentence | `Expression` | [`src/logic.jl`][logic] |
| 7.10    | TT-Entails         | `tt_entails` | [`src/logic.jl`][logic] |
| 7.12    | PL-Resolution      | `pl_resolution` | [`src/logic.jl`][logic] |
| 7.14    | Convert to CNF     | `to_conjunctive_normal_form` | [`src/logic.jl`][logic] |
| 7.15    | PL-FC-Entails?     | `pl_fc_resolution` | [`src/logic.jl`][logic] |
| 7.17    | DPLL-Satisfiable?  | `dpll_satisfiable` | [`src/logic.jl`][logic] |
| 7.18    | WalkSAT            | `walksat` | [`src/logic.jl`][logic] |
| 7.20    | Hybrid-Wumpus-Agent |  |  |
| 7.22    | SATPlan            | `sat_plan`  | [`src/logic.jl`][logic] |
| 9       | Subst              | `substitute` | [`src/logic.jl`][logic] |
| 9.1     | Unify              | `unify` | [`src/logic.jl`][logic] |
| 9.3     | FOL-FC-Ask         | `fol_fc_ask` | [`src/logic.jl`][logic] |
| 9.6     | FOL-BC-Ask         | `fol_bc_ask` | [`src/logic.jl`][logic] |
| 9.8     | Append             |  |  |
| 10.1    | Air-Cargo-problem  | `air_cargo_pddl` |[`src/planning.jl`][planning]|
| 10.2    | Spare-Tire-Problem | `spare_tire_pddl` |[`src/planning.jl`][planning]|
| 10.3    | Three-Block-Tower  | `three_block_tower_pddl` |[`src/planning.jl`][planning]|
| 10.7    | Cake-Problem       | `have_cake_and_eat_cake_too_pddl` |[`src/planning.jl`][planning]|
| 10.9    | Graphplan          | `graphplan` | [`src/planning.jl`][planning] |
| 10.13   | Partial-Order-Planner |  |  |
| 11.1    | Job-Shop-Problem-With-Resources | `job_shop_scheduling_pddl` |[`src/planning.jl`][planning]|
| 11.5    | Hierarchical-Search | `hierarchical_search` |[`src/planning.jl`][planning]|
| 11.8    | Angelic-Search   |  |  |
| 11.10   | Doubles-tennis     | `doubles_tennis_pddl` | [`src/planning.jl`][planning] |
| 13      | Discrete Probability Distribution | `ProbDist` | [`src/probability.jl`][probability] |
| 13.1    | DT-Agent                    | `DecisionTheoreticAgentProgram` | [`src/probability.jl`][probability] |
| 14.9    | Enumeration-Ask             |  |  |
| 14.11   | Elimination-Ask             |  |  |
| 14.13   | Prior-Sample                |  |  |
| 14.14   | Rejection-Sampling          |  |  |
| 14.15   | Likelihood-Weighting        |  |  |
| 14.16   | Gibbs-Ask                   |  |  |
| 15.4    | Forward-Backward            |  |  |
| 15.6    | Fixed-Lag-Smoothing         |  |  |
| 15.17   | Particle-Filtering          |  |  |
| 16.9    | Information-Gathering-Agent |  |  |
| 17.4    | Value-Iteration             |  |  |
| 17.7    | Policy-Iteration            |  |  |
| 17.7    | POMDP-Value-Iteration  	    |  |  |
| 18.5    | Decision-Tree-Learning 	    |  |  |
| 18.8    | Cross-Validation       	    |  |  |
| 18.11   | Decision-List-Learning 	    |  |  |
| 18.24   | Back-Prop-Learning     	    |  |  |
| 18.34   | AdaBoost               	    |  |  |
| 19.2    | Current-Best-Learning  	    |  |  |
| 19.3    | Version-Space-Learning 	    |  |  |
| 19.8    | Minimal-Consistent-Det 	    |  |  |
| 19.12   | FOIL               |  |  |
| 21.2    | Passive-ADP-Agent  |  |  |
| 21.4    | Passive-TD-Agent   |  |  |
| 21.8    | Q-Learning-Agent   |  |  |
| 22.1    | HITS               |  |  |
| 23      | Chart-Parse        |  |  |
| 23.5    | CYK-Parse          |  |  |
| 25.9    | Monte-Carlo-Localization|  |  |

Running tests
-------------

All Base.Test tests for the aima-julia project can be found in the [tests](https://github.com/aimacode/aima-julia/tree/master/tests) directory.

Conventions
-----------

* 4 spaces, not tabs

* Please try to follow the style conventions of the file your are modifying.

We like this [style guide](https://docs.julialang.org/en/release-0.5/manual/style-guide/).

## Acknowledgements

The algorithms implemented in this project are found from both Russell And Norvig's "Artificial Intelligence - A Modern Approach" and [aima-pseudocode](https://github.com/aimacode/aima-pseudocode).