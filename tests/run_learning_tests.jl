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

nbdm = NaiveBayesLearner(iris_dataset, continuous=false);

@test (predict(nbdm, [5, 3, 1, 0.1]) == "setosa");

@test (predict(nbdm, [6, 3, 4, 1.1]) == "versicolor");

@test (predict(nbdm, [7.7, 3, 6, 2]) == "virginica");

nbcm = NaiveBayesLearner(iris_dataset, continuous=true);

@test (predict(nbcm, [5, 3, 1, 0.1]) == "setosa");

@test (predict(nbcm, [6, 5, 3, 1.5]) == "versicolor");

@test (predict(nbcm, [7, 3, 6.5, 2]) == "virginica");

iris_dataset = DataSet(name="iris", examples="./aima-data/iris.csv");

k_nearest_neighbors = NearestNeighborLearner(iris_dataset, 3);

@test (predict(k_nearest_neighbors, [5, 3, 1, 0.1]) == "setosa");

@test (predict(k_nearest_neighbors, [6, 5, 3, 1.5]) == "versicolor");

@test (predict(k_nearest_neighbors, [7.5, 4, 6, 2]) == "virginica");

