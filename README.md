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
| 2.1     | Environment        | `Environment` | [`agents.jl`](agents.jl) |
| 2.1     | Agent              | `Agent` | [`agents.jl`](agents.jl) |
| 2.3     | Table-Driven-Vacuum-Agent | `TableDrivenVacuumAgent` | [`agents.jl`](agents.jl) |
| 2.7     | Table-Driven-Agent | `TableDrivenAgentProgram` | [`agents.jl`](agents.jl) |
| 2.8     | Reflex-Vacuum-Agent | `ReflexVacuumAgent` | [`agents.jl`](agents.jl) |
| 2.10    | Simple-Reflex-Agent | `SimpleReflexAgent` | [`agents.jl`](agents.jl) |
| 2.12    | Model-Based-Reflex-Agent | `ModelBasedReflexAgentProgram` | [`agents.jl`](agents.jl) |
| 3       | Problem            | `Problem` | [`search.jl`](search.jl) |
| 3       | Node               | `Node` | [`search.jl`](search.jl) |
| 3       | Queue              | `Queue` | [`utils.jl`](utils.jl) |
| 3.1     | Simple-Problem-Solving-Agent | `SimpleProblemSolvingAgent` | [`search.jl`](search.jl) |
| 3.2     | Romania            | `romania` | [`search.jl`](search.jl) |
| 3.7     | Tree-Search        | `tree_search` | [`search.jl`](search.jl) |
| 3.7     | Graph-Search        | `graph_search` | [`search.jl`](search.jl) |
| 3.11    | Breadth-First-Search        | `breadth_first_search` | [`search.jl`](search.jl) |
| 3.14    | Uniform-Cost-Search        | `uniform_cost_search` | [`search.jl`](search.jl) |
| 3.17    | Depth-Limited-Search | `depth_limited_search` | [`search.jl`](search.jl) |
| 3.18    | Iterative-Deepening-Search | `iterative_deepening_search` | [`search.jl`](search.jl) |
| 3.22    | Best-First-Search  | `best_first_graph_search` | [`search.jl`](search.jl) |
| 3.24    | A\*-Search        | `astar_search` | [`search.jl`](search.jl) |
| 3.26    | Recursive-Best-First-Search | `recursive_best_first_search` | [`search.jl`](search.jl) |
| 4.2     | Hill-Climbing      | `hill_climbing` | [`search.jl`](search.jl) |
| 4.5     | Simulated-Annealing | `simulated_annealing` | [`search.jl`](search.jl) |
| 4.8     | Genetic-Algorithm  | `genetic_algorithm` | [`search.jl`](search.jl) |
| 4.11    | And-Or-Graph-Search | `and_or_graph_search` | [`search.jl`](search.jl) |
| 4.21    | Online-DFS-Agent   | `OnlineDFSAgentProgram` | [`search.jl`](search.jl) |
| 4.24    | LRTA\*-Agent       | `LRTAStarAgentProgram` | [`search.jl`](search.jl) |
| 5.3     | Minimax-Decision   | `minimax_decision` | [`games.jl`](games.jl) |
| 5.7     | Alpha-Beta-Search  | `alphabeta_search` | [`games.jl`](games.jl) |
| 6       | CSP                | `CSP` | [`csp.jl`](csp.jl) |
| 6.3     | AC-3               | `AC3` | [`csp.jl`](csp.jl) |
| 6.5     | Backtracking-Search | `backtracking_search` | [`csp.jl`](csp.jl) |
| 6.8     | Min-Conflicts      | `min_conflicts` | [`csp.jl`](csp.jl) |
| 6.11    | Tree-CSP-Solver    | `tree_csp_solver` | [`csp.jl`](csp.jl) |
| 7       | KB                 | `KnowledgeBase` | [`logic.jl`](logic.jl) |
| 7.1     | KB-Agent           | `KnowledgeBaseAgentProgram` | [`logic.jl`](logic.jl) |
| 7.7     | Propositional Logic Sentence | `Expression` | [`logic.jl`](logic.jl) |
| 7.10    | TT-Entails         | `tt_entails` | [`logic.jl`](logic.jl) |
| 7.12    | PL-Resolution      | `pl_resolution` | [`logic.jl`](logic.jl) |
| 7.14    | Convert to CNF     | `to_conjunctive_normal_form` | [`logic.jl`](logic.jl) |
| 7.15    | PL-FC-Entails?     | `pl_fc_resolution` | [`logic.jl`](logic.jl) |
| 7.17    | DPLL-Satisfiable?  | `dpll_satisfiable` | [`logic.jl`](logic.jl) |
| 7.18    | WalkSAT            | `walksat` | [`logic.jl`](logic.jl) |
| 7.20    | Hybrid-Wumpus-Agent |  |  |
| 7.22    | SATPlan            | `sat_plan`  | [`logic.jl`](logic.jl) |
| 9       | Subst              | `substitute` | [`logic.jl`](logic.jl) |
| 9.1     | Unify              | `unify` | [`logic.jl`](logic.jl) |
| 9.3     | FOL-FC-Ask         | `fol_fc_ask` | [`logic.jl`](logic.jl) |
| 9.6     | FOL-BC-Ask         | `fol_bc_ask` | [`logic.jl`](logic.jl) |
| 9.8     | Append             |  |  |
| 10.1    | Air-Cargo-problem  | `air_cargo_pddl` |[`planning.jl`](planning.jl) |
| 10.2    | Spare-Tire-problem | `spare_tire_pddl` |[`planning.jl`](planning.jl) |
| 10.3    | Three-Block-Tower  | `three_block_tower_pddl` |[`planning.jl`](planning.jl) |
| 10.7    | Cake-problem       | `have_cake_and_eat_cake_too_pddl` |[`planning.jl`](planning.jl) |
| 10.9    | Graphplan          | `graphplan` | [`planning.jl`](planning.jl) |
| 10.13   | Partial-Order-Planner |  |  |
| 11.1    | Job-Shop-Problem-With-Resources | `job_shop_scheduling_pddl` |[`planning.jl`](planning.jl) |
| 11.5    | Hierarchical-Search | `hierarchical_search` | [`planning.jl`](planning.jl) |
| 11.8    | Angelic-Search   |  |  |
| 11.10   | Doubles-Tennis-problem   | `doubles_tennis_pddl` | [`planning.jl`](planning.jl) |
| 13      | Discrete Probability Distribution | `ProbabilityDistribution` | [`probability.jl`](probability.jl) |
| 13.1    | DT-Agent                    | `DecisionTheoreticAgentProgram` | [`probability.jl`](probability.jl) |
| 14.9    | Enumeration-Ask             | `enumeration_ask` | [`probability.jl`](probability.jl) |
| 14.11   | Elimination-Ask             | `elimination_ask` | [`probability.jl`](probability.jl) |
| 14.13   | Prior-Sample                | `prior_sample` | [`probability.jl`](probability.jl) |
| 14.14   | Rejection-Sampling          | `rejection_sample` | [`probability.jl`](probability.jl) |
| 14.15   | Likelihood-Weighting        | `likelihood_weighting` | [`probability.jl`](probability.jl) |
| 14.16   | Gibbs-Ask                   | `gibbs_ask` | [`probability.jl`](probability.jl) |
| 15.4    | Forward-Backward            | `forward_backward` | [`probability.jl`](probability.jl) |
| 15.6    | Fixed-Lag-Smoothing         | `fixed_lag_smoothing` | [`probability.jl`](probability.jl) |
| 15.17   | Particle-Filtering          | `particle_filtering` | [`probability.jl`](probability.jl) |
| 16.9    | Information-Gathering-Agent |  |  |
| 17.4    | Value-Iteration             | `value_iteration` | [`mdp.jl`](mdp.jl) |
| 17.7    | Policy-Iteration            | `policy_iteration` | [`mdp.jl`](mdp.jl) |
| 17.7    | POMDP-Value-Iteration  	    |  |  |
| 18.5    | Decision-Tree-Learning 	    | `decision_tree_learning` | [`learning.jl`](learning.jl) |
| 18.8    | Cross-Validation       	    | `cross_validation` | [`learning.jl`](learning.jl) |
| 18.11   | Decision-List-Learning 	    | `decision_list_learning` | [`learning.jl`](learning.jl) |
| 18.24   | Back-Prop-Learning     	    | `back_propagation_learning!` | [`learning.jl`](learning.jl) |
| 18.34   | AdaBoost               	    | `adaboost!` | [`learning.jl`](learning.jl) |
| 19.2    | Current-Best-Learning  	    | `current_best_learning` | [`kl.jl`](kl.jl) |
| 19.3    | Version-Space-Learning 	    | `version_space_learning` | [`kl.jl`](kl.jl) |
| 19.8    | Minimal-Consistent-Det 	    | `minimal_consistent_determination` | [`kl.jl`](kl.jl) |
| 19.12   | FOIL               | `foil` | [`kl.jl`](kl.jl) |
| 21.2    | Passive-ADP-Agent  | `PassiveADPAgentProgram` | [`rl.jl`](rl.jl) |
| 21.4    | Passive-TD-Agent   | `PassiveTDAgentProgram` | [`rl.jl`](rl.jl) |
| 21.8    | Q-Learning-Agent   | `QLearningAgentProgram` | [`rl.jl`](rl.jl) |
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

* Please try to follow the style conventions of the file you are modifying.

We like this [style guide](https://docs.julialang.org/en/release-0.5/manual/style-guide/).

## Acknowledgements

The algorithms implemented in this project are found from both Russell And Norvig's "Artificial Intelligence - A Modern Approach" and [aima-pseudocode](https://github.com/aimacode/aima-pseudocode).