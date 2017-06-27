# aima-julia

[![Build Status](https://travis-ci.org/aimacode/aima-julia.svg?branch=master)](https://travis-ci.org/aimacode/aima-julia)

Julia (v0.5+) implementation of the algorithms found in "Artificial Intelligence: A Modern Approach".

## Acknowledgements

The algorithms implemented in this project are found from both Russell And Norvig's "Artificial Intelligence - A Modern Approach" and [aima-pseudocode](https://github.com/aimacode/aima-pseudocode).

## Intent

The intent of this fork is to align the code with a clearer interface like structure which is there in the aima-java like structure.

### Index of Implemented Algorithms

|Fig       |Page      |Name (in book)|Code    |File   |
| -------- |:--------:| :------------| :----- |:------|
|2|34|Environment|[Environment]|[agent.jl](/agents.jl)|
|2.1|35|Agent|[Agent]|[agent.jl](/agents.jl)|
|2.3|36|Table-Driven-Vacuum-Agent|[TableDrivenVacuumAgent]|[agent.jl](/agents.jl)|
|2.7|47|Table-Driven-Agent|[TableDrivenAgentProgram]|[agent.jl](/agents.jl)|
|2.8|48|Reflex-Vacuum-Agent|[ReflexVacuumAgent]|[agent.jl](/agents.jl)|
|2.10|49|Simple-Reflex-Agent|[SimpleReflexAgentProgram]|[agent.jl](/agents.jl)|
|2.12|51|Model-Based-Reflex-Agent|[ModelBasedReflexAgentProgram]|[agent.jl](/agents.jl)|
|3|66|Problem|[Problem]|
|3.1|67|Simple-Problem-Solving-Agent|[SimpleProblemSolvingAgent]|
|3.2|68|Romania|[SimplifiedRoadMapOfPartOfRomania]|
|3.7|77|Tree-Search|[TreeSearch]|
|3.7|77|Graph-Search|[GraphSearch]|
|3.10|79|Node|[Node]|
|3.11|82|Breadth-First-Search|[BreadthFirstSearch]|
|3.14|84|Uniform-Cost-Search|[UniformCostSearch]|
|3|85|Depth-first Search|[DepthFirstSearch]|
|3.17|88|Depth-Limited-Search|[DepthLimitedSearch]|
|3.18|89|Iterative-Deepening-Search|[IterativeDeepeningSearch]|
|3|90|Bidirectional search|[BidirectionalSearch]|
|3|92|Best-First search|[BestFirstSearch]|
|3|92|Greedy best-First search|[GreedyBestFirstSearch]|
|3|93|A\* Search|[AStarSearch]|
|3.26|99|Recursive-Best-First-Search |[RecursiveBestFirstSearch]|
|4.2|122|Hill-Climbing|[HillClimbingSearch]|
|4.5|126|Simulated-Annealing|[SimulatedAnnealingSearch]|
|4.8|129|Genetic-Algorithm|[GeneticAlgorithm]|
|4.11|136|And-Or-Graph-Search|[AndOrSearch]|
|4|147|Online search problem|[OnlineSearchProblem]|
|4.21|150|Online-DFS-Agent|[OnlineDFSAgent]|
|4.24|152|LRTA\*-Agent|[LRTAStarAgent]|
|5.3|166|Minimax-Decision|[MinimaxSearch]|
|5.7|170|Alpha-Beta-Search|[AlphaBetaSearch]|
|6|202|CSP|[CSP]|
|6.1|204|Map CSP|[MapCSP]|
|6.3|209|AC-3|[AC3Strategy]|
|6.5|215|Backtracking-Search|[BacktrackingStrategy]|
|6.8|221|Min-Conflicts|[MinConflictsStrategy]|
|6.11|224|Tree-CSP-Solver|[TreeCSPSolver]|
|7|235|Knowledge Base|[KnowledgeBase]|
|7.1|236|KB-Agent|[KBAgent]|
|7.7|244|Propositional-Logic-Sentence|[Sentence]|
|7.10|248|TT-Entails|[TTEntails]|
|7|253|Convert-to-CNF|[ConvertToCNF]|
|7.12|255|PL-Resolution|[PLResolution]|
|7.15|258|PL-FC-Entails?|[PLFCEntails]|
|7.17|261|DPLL-Satisfiable?|[DPLLSatisfiable]|
|7.18|263|WalkSAT|[WalkSAT]|
|7.20|270|Hybrid-Wumpus-Agent|[HybridWumpusAgent]|
|7.22|272|SATPlan|[SATPlan]|
|9|323|Subst|[SubstVisitor]|
|9.1|328|Unify|[Unifier]|
|9.3|332|FOL-FC-Ask|[FOLFCAsk]|
|9.3|332|FOL-BC-Ask|[FOLBCAsk]|
|9|345|CNF|[CNFConverter]|
|9|347|Resolution|[FOLTFMResolution]|
|9|354|Demodulation|[Demodulation]|
|9|354|Paramodulation|[Paramodulation]|
|9|345|Subsumption|[SubsumptionElimination]|
|10.9|383|Graphplan|---|
|11.5|409|Hierarchical-Search|---|
|11.8|414|Angelic-Search|---|
|13.1|484|DT-Agent|---|
|13|484|Probability-Model|[ProbabilityModel]|
|13|487|Probability-Distribution|[ProbabilityDistribution]|
|13|490|Full-Joint-Distribution|[FullJointDistributionModel]|
|14|510|Bayesian Network|[BayesianNetwork]|
|14.9|525|Enumeration-Ask|[EnumerationAsk]|
|14.11|528|Elimination-Ask|[EliminationAsk]|
|14.13|531|Prior-Sample|[PriorSample]|
|14.14|533|Rejection-Sampling|[RejectionSampling]|
|14.15|534|Likelihood-Weighting|[LikelihoodWeighting]|
|14.16|537|GIBBS-Ask|[GibbsAsk]|
|15.4|576|Forward-Backward|[ForwardBackward]|
|15|578|Hidden Markov Model|[HiddenMarkovModel]|
|15.6|580|Fixed-Lag-Smoothing|[FixedLagSmoothing]|
|15|590|Dynamic Bayesian Network|[DynamicBayesianNetwork]|
|15.17|598|Particle-Filtering|[ParticleFiltering]|
|16.9|632|Information-Gathering-Agent|---|
|17|647|Markov Decision Process|[MarkovDecisionProcess]|
|17.4|653|Value-Iteration|[ValueIteration]|
|17.7|657|Policy-Iteration|[PolicyIteration]|
|17.9|663|POMDP-Value-Iteration|---|
|18.5|702|Decision-Tree-Learning|[DecisionTreeLearner]|
|18.8|710|Cross-Validation-Wrapper|---|
|18.11|717|Decision-List-Learning|[DecisionListLearner]|
|18.24|734|Back-Prop-Learning|[BackPropLearning]|
|18.34|751|AdaBoost|[AdaBoostLearner]|
|19.2|771|Current-Best-Learning|---|
|19.3|773|Version-Space-Learning|---|
|19.8|786|Minimal-Consistent-Det|---|
|19.12|793|FOIL|---|
|21.2|834|Passive-ADP-Agent|[PassiveADPAgent]|
|21.4|837|Passive-TD-Agent|[PassiveTDAgent]|
|21.8|844|Q-Learning-Agent|[QLearningAgent]|
|22.1|871|HITS|[HITS]|
|23.5|894|CYK-Parse|[CYK]|
|25.9|982|Monte-Carlo-Localization|[MonteCarloLocalization]|
