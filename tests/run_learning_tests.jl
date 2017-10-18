include("../aimajulia.jl");

using Base.Test;

using aimajulia;

using aimajulia.utils;

#The following learning tests are from the aima-python doctests

@test (repr(euclidean_distance([1, 2], [3, 4])) == "2.8284271247461903");

@test (repr(euclidean_distance([1, 2, 3], [4, 5, 6])) == "5.196152422706632");

@test (repr(euclidean_distance([0, 0, 0], [0, 0, 0])) == "0.0");

@test (root_mean_square_error([2, 2], [2, 2]) == 0);

@test (root_mean_square_error([0, 0], [0, 1]) == sqrt(0.5));

@test (root_mean_square_error([1, 0], [0, 1]) == 1);

@test (root_mean_square_error([0, 0], [0, -1]) == sqrt(0.5));

@test (root_mean_square_error([0, 0.5], [0, -0.5]) == sqrt(0.5));

@test (manhattan_distance([2, 2], [2, 2]) == 0);

@test (manhattan_distance([0, 0], [0, 1]) == 1);

@test (manhattan_distance([1, 0], [0, 1]) == 2);

@test (manhattan_distance([0, 0], [0, -1]) == 1);

@test (manhattan_distance([0, 0.5], [0, -0.5]) == 1);

@test (mean_boolean_error([1, 1], [0, 0]) == 1)

@test (mean_boolean_error([0, 1], [1, 0]) == 1)

@test (mean_boolean_error([1, 1], [0, 1]) == 0.5)

@test (mean_boolean_error([0, 0], [0, 0]) == 0)

@test (mean_boolean_error([1, 1], [1, 1]) == 0)

@test (mean_error([2, 2], [2, 2]) == 0);

@test (mean_error([0, 0], [0, 1]) == 0.5);

@test (mean_error([1, 0], [0, 1]) ==  1);

@test (mean_error([0, 0], [0, -1]) ==  0.5);

@test (mean_error([0, 0.5], [0, -0.5]) == 0.5);

@test (gaussian(1,0.5,0.7) == 0.6664492057835993);

@test (gaussian(5,2,4.5) == 0.19333405840142462);

@test (gaussian(3,1,3) == 0.3989422804014327);

iris_dataset = DataSet(name="iris", examples="./aima-data/iris.csv", exclude=[4]);

@test (iris_dataset.inputs == [1, 2, 3]);

iris_dataset = DataSet(name="iris", examples="./aima-data/iris.csv");
means_dict, deviations_dict = find_means_and_deviations(iris_dataset);

@test (means_dict["setosa"][1] == 5.006);

@test (means_dict["versicolor"][1] == 5.936);

@test (means_dict["virginica"][1] == 6.587999999999999);

@test (deviations_dict["setosa"][1] == 0.3524896872134513);

@test (deviations_dict["versicolor"][1] == 0.5161711470638634);

@test (deviations_dict["virginica"][1] == 0.6358795932744321);

cpd = CountingProbabilityDistribution();

for i in 1:10000
    add(cpd, rand(RandomDeviceInstance, ["1", "2", "3", "4", "5", "6"]));
end

probabilities = collect(cpd[n] for n in ("1", "2", "3", "4", "5", "6"));

@test ((1.0/7.0) <= reduce(min, probabilities) <= reduce(max, probabilities) <= (1.0/5.0));

zoo_dataset = DataSet(name="zoo", examples="./aima-data/zoo.csv");

pl = PluralityLearner(zoo_dataset);

@test (predict(pl, [1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 4, 1, 0, 1]) == "mammal");

iris_dataset = DataSet(name="iris", examples="./aima-data/iris.csv");

# Naive Discrete Model
nbdm = NaiveBayesLearner(iris_dataset, continuous=false);

@test (predict(nbdm, [5, 3, 1, 0.1]) == "setosa");

@test (predict(nbdm, [6, 3, 4, 1.1]) == "versicolor");

@test (predict(nbdm, [7.7, 3, 6, 2]) == "virginica");

# Naive Continuous Model
nbcm = NaiveBayesLearner(iris_dataset, continuous=true);

@test (predict(nbcm, [5, 3, 1, 0.1]) == "setosa");

@test (predict(nbcm, [6, 5, 3, 1.5]) == "versicolor");

@test (predict(nbcm, [7, 3, 6.5, 2]) == "virginica");

# Naive Conditional Probability Model
d1 = CountingProbabilityDistribution(vcat(fill('a', 50), fill('b', 30), fill('c', 15)));
d2 = CountingProbabilityDistribution(vcat(fill('a', 30), fill('b', 45), fill('c', 20)));
d3 = CountingProbabilityDistribution(vcat(fill('a', 20), fill('b', 20), fill('c', 35)));
nbsm = NaiveBayesLearner(Dict([(("First", 0.5), d1), (("Second", 0.3), d2), (("Third", 0.2), d3)]), simple=true);

@test (predict(nbsm, "aab") == "First");

@test (predict(nbsm, ['b', 'b']) == "Second");

@test (predict(nbsm, "ccbcc") == "Third");

iris_dataset = DataSet(name="iris", examples="./aima-data/iris.csv");

k_nearest_neighbors = NearestNeighborLearner(iris_dataset, 3);

@test (predict(k_nearest_neighbors, [5, 3, 1, 0.1]) == "setosa");

@test (predict(k_nearest_neighbors, [6, 5, 3, 1.5]) == "versicolor");

@test (predict(k_nearest_neighbors, [7.5, 4, 6, 2]) == "virginica");

iris_dataset = DataSet(name="iris", examples="./aima-data/iris.csv");

dtl = DecisionTreeLearner(iris_dataset);

@test (predict(dtl, [5, 3, 1, 0.1]) == "setosa");

@test (predict(dtl, [6, 5, 3, 1.5]) == "versicolor");

@test (predict(dtl, [7.5, 4, 6, 2]) == "virginica");

function test_rf_predictions(ex1_results::AbstractVector, ex2_results::AbstractVector, ex3_results::AbstractVector)
    local rf::RandomForest = RandomForest(iris_dataset);
    push!(ex1_results, (predict(rf, [5, 3, 1, 0.1]) == "setosa"));
    push!(ex2_results, (predict(rf, [6, 5, 3, 1]) == "versicolor"));
    push!(ex3_results, (predict(rf, [7.5, 4, 6, 2]) == "virginica"));
    nothing;
end

setosa_results = Array{Bool, 1}();
versicolor_results = Array{Bool, 1}();
virginica_results = Array{Bool, 1}();
# Run test_rf_predictions() 1000 times.
println("@time for i in 1:1000\n\ttest_rf_predictions(setosa_results, versicolor_results, virginica_results);\nend");
@time for i in 1:1000
    test_rf_predictions(setosa_results, versicolor_results, virginica_results);
end

setosa_results_count = count((function(b::Bool)
                                    return b;
                                end), setosa_results);
versicolor_results_count = count((function(b::Bool)
                                    return b;
                                end), versicolor_results);
virginica_results_count = count((function(b::Bool)
                                    return b;
                                end), virginica_results);

# lowest setosa_results_count result previously obtained was 970
@test (setosa_results_count >= 960);

# lowest versicolor_results_count result previously obtained was 959
@test (versicolor_results_count >= 950);

# lowest virginica_results_count result previously obtained was 996
@test (virginica_results_count >= 990);

println();
println("setosa assert count (out of 1000): ", setosa_results_count);
println("setosa assertion failure rate: approximately ", Float64(1000 - setosa_results_count)/10.0, "%");
println("versicolor assert count (out of 1000): ", versicolor_results_count);
println("versicolor assertion failure rate: approximately ", Float64(1000 - versicolor_results_count)/10.0, "%");
println("virginica assert count (out of 1000): ", virginica_results_count);
println("virginica assertion failure rate: approximately ", Float64(1000 - virginica_results_count)/10.0, "%");
println();

weights = random_weights(-0.5, 0.5, 10);

@test (length(weights) == 10);

@test (all(((weight >= -0.5) && (weight <= 0.5)) for weight in weights));

iris_dataset = DataSet(name="iris", examples="./aima-data/iris.csv");

# The DataType of the example classification must match the eltype of the classes array.

classes = map(SubString{String}, ["setosa", "versicolor", "virginica"]);

classes_to_numbers(iris_dataset, classes);

nnl = NeuralNetworkLearner(iris_dataset, hidden_layers_sizes=[5], learning_rate=0.15, epochs=75);

neural_network_learner_score = aimajulia.grade_learner(nnl,
                                                        [([5, 3, 1, 0.1], 1),
                                                        ([5, 3.5, 1, 0], 1),
                                                        ([6, 3, 4, 1.1], 2),
                                                        ([6, 2, 3.5, 1], 2),
                                                        ([7.5, 4, 6, 2], 3),
                                                        ([7, 3, 6, 2.5], 3)]);

println("neural network learner score (out of 1.0): ", neural_network_learner_score);

# Allow up to 2 failed tests.
@test (neural_network_learner_score >= (2/3));

neural_network_learner_error_ratio = aimajulia.error_ratio(nnl, iris_dataset);

println("neural network learner error ratio: ", (neural_network_learner_error_ratio * 100), "%");
println();

# NeuralNetworkLearner previously had an error ratio of 0.33333333333333337.
@test (neural_network_learner_error_ratio < 0.40);

iris_dataset = DataSet(name="iris", examples="./aima-data/iris.csv");

classes_to_numbers(iris_dataset, nothing);

pl = PerceptronLearner(iris_dataset);

perceptron_learner_score = aimajulia.grade_learner(pl,
                                                    [([5, 3, 1, 0.1], 1),
                                                    ([5, 3.5, 1, 0], 1),
                                                    ([6, 3, 4, 1.1], 2),
                                                    ([6, 2, 3.5, 1], 2),
                                                    ([7.5, 4, 6, 2], 3),
                                                    ([7, 3, 6, 2.5], 3)]);

println("perceptron learner score (out of 1.0): ", perceptron_learner_score);

# Allow up to 3 failed tests.
@test (perceptron_learner_score >= (1/2));

perceptron_learner_error_ratio = aimajulia.error_ratio(pl, iris_dataset);

println("perceptron learner error ratio: ", (perceptron_learner_error_ratio * 100), "%");
println();

@test (perceptron_learner_error_ratio < 0.40);

@test (weighted_mode("abbaa", [1, 2, 3, 1, 2]) == "b");

@test (weighted_mode(["a", "b", "b", "a", "a"], [1, 2, 3, 1, 2]) == "b");

@test (weighted_replicate(["A", "B", "C"], [1, 2, 1], 4) == ["A", "B", "B", "C"]);

