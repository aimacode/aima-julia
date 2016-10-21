<div align="center">
  <a href="http://aima.cs.berkeley.edu/"><img src="https://raw.githubusercontent.com/NirantK/aima-julia/master/images/aima_logo.png"></a><br><br>
</div>
----------------
# aima-julia
Julia implementation of algorithms from Russell And Norvig's "Artificial Intelligence - A Modern Approach"

## Julia 0.5
This repository uses *Julia 0.5* Stable Release. You can download and install the latest [Julia version](https://www.julialang.org/downloads). Alternatively, use a browser-based Julia interpreter such as [JuliaBox](https://www.juliabox.com/). 

## Structure of the Project

The structure of this project is the same as [Python Code for AIMA](https://github.com/aimacode/aima-python). 

When complete, this project will have Julia code for all the pseudocode algorithms in the book. For each major topic, such as `logic`, we will have the following three files in the main branch:

- `logic.jl`: Implementations of all the pseudocode algorithms, and necessary support functions/classes/data.
- `logic.ipynb`: A Jupyter (IJulia) notebook that explains and gives examples of how to use the code.
- `tests/logic_test.jl`: A lightweight test suite, using `assert` statements

## Index of Code

It would be nice to have a table of algorithms, the figure, name of the code in the book and in the repository, and the file where they are implemented in the code. This chart follows third edition of the book. As you see, we are just starting the Julia repository. Feel free to pick any section and contribute. 


| **Figure** | **Name (in 3<sup>rd</sup> edition)** | **Name (in repository)** | **File**
|:--------|:-------------------|:---------|:-----------|
| 2.1     | Environment        |          |           |
| 2.1     | Agent              |          |           |
| 2.3     | Table-Driven-Vacuum-Agent |          |           |
| 2.7     | Table-Driven-Agent |          |           |
| 2.8     | Reflex-Vacuum-Agent |          |           |
| 2.10    | Simple-Reflex-Agent |          |           |
| 2.12    | Model-Based-Reflex-Agent |          |           |
| 3       | Problem            |          |           |
| 3       | Node               |          |           |
| 3       | Queue              |          |           |
| 3.1     | Simple-Problem-Solving-Agent |          |           |
| 3.2     | Romania            |          |           |
| 3.7     | Tree-Search        |          |           |
| 3.7     | Graph-Search        |         |           |
| 3.11    | Breadth-First-Search        |          |           |
| 3.14    | Uniform-Cost-Search        |          |           |
| 3.17    | Depth-Limited-Search |          |           |
| 3.18    | Iterative-Deepening-Search |          |           |
| 3.22    | Best-First-Search  |         |           |
| 3.24    | A\*-Search        |          |           |
| 3.26    | Recursive-Best-First-Search |          |           |
| 4.2     | Hill-Climbing      |          |           |
| 4.5     | Simulated-Annealing |          |           |
| 4.8     | Genetic-Algorithm  |          |           |
| 4.11    | And-Or-Graph-Search |          |           |
| 4.21    | Online-DFS-Agent   |          |           |
| 4.24    | LRTA\*-Agent       |          |           |
| 5.3     | Minimax-Decision   |          |           |
| 5.7     | Alpha-Beta-Search  |        |           |
| 6       | CSP                |          |           |
| 6.3     | AC-3               |          |           |
| 6.5     | Backtracking-Search |          |           |
| 6.8     | Min-Conflicts      |          |           |
| 6.11    | Tree-CSP-Solver    |          |           |
| 7       | KB                 |          |           |
| 7.1     | KB-Agent           |          |           |
| 7.7     | Propositional Logic Sentence |          |           |
| 7.10    | TT-Entails         |          |           |
| 7.12    | PL-Resolution      |          |           |
| 7.14    | Convert to CNF     |         |           |
| 7.15    | PL-FC-Entails?     |          |           |
| 7.17    | DPLL-Satisfiable?  |          |           |
| 7.18    | WalkSAT            |          |           |
| 7.20    | Hybrid-Wumpus-Agent    |         |           |
| 7.22    | SATPlan            |          |           |
| 9       | Subst              |          |           |
| 9.1     | Unify              |          |           |
| 9.3     | FOL-FC-Ask         |          |           |
| 9.6     | FOL-BC-Ask         |          |           |
| 9.8     | Append             |            |              |
| 10.1    | Air-Cargo-problem    |          |
| 10.2    | Spare-Tire-Problem |          |
| 10.3    | Three-Block-Tower  |          |
| 10.7    | Cake-Problem       |          |
| 10.9    | Graphplan          |          |
| 10.13   | Partial-Order-Planner |          |
| 11.1    | Job-Shop-Problem-With-Resources |          |
| 11.5    | Hierarchical-Search |          |
| 11.8    | Angelic-Search   |          |
| 11.10   | Doubles-tennis     |          |
| 13      | Discrete Probability Distribution |          |           |
| 13.1    | DT-Agent           |          |           |
| 14.9    | Enumeration-Ask    |          |           |
| 14.11   | Elimination-Ask    |          |           |
| 14.13   | Prior-Sample       |          |           |
| 14.14   | Rejection-Sampling |          |           |
| 14.15   | Likelihood-Weighting |          |           |
| 14.16   | Gibbs-Ask           |          |           |
| 15.4    | Forward-Backward   |          |           |
| 15.6    | Fixed-Lag-Smoothing |          |           |
| 15.17   | Particle-Filtering |          |           |
| 16.9    | Information-Gathering-Agent |          |          |           |
| 17.4    | Value-Iteration    |          |           |
| 17.7    | Policy-Iteration   |          |           |
| 17.7    | POMDP-Value-Iteration  |           |        |
| 18.5    | Decision-Tree-Learning |          |           |
| 18.8    | Cross-Validation   |          |           |
| 18.11   | Decision-List-Learning |          |           |
| 18.24   | Back-Prop-Learning |          |           |
| 18.34   | AdaBoost           |          |           |
| 19.2    | Current-Best-Learning |          |
| 19.3    | Version-Space-Learning |          |
| 19.8    | Minimal-Consistent-Det |          |
| 19.12   | FOIL               |          |
| 21.2    | Passive-ADP-Agent  |          |           |
| 21.4    | Passive-TD-Agent   |          |           |
| 21.8    | Q-Learning-Agent   |          |           |
| 22.1    | HITS               |         |         |
| 23      | Chart-Parse        |          |           |
| 23.5    | CYK-Parse          |          |           |
| 25.9    | Monte-Carlo-Localization|       |

## Index of data structures

| **Figure** | **Name (in repository)** | **File** |
|:-----------|:-------------------------|:---------|
| 3.2    | romania_map              | |
| 4.9    | vacumm_world             | |
| 4.23   | one_dim_state_space      | |
| 6.1    | australia_map            | |
| 7.13   | wumpus_world_inference   | |
| 7.16   | horn_clauses_KB          | |
| 17.1   | sequential_decision_environment | |
| 18.2   | waiting_decision_tree    | |