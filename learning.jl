
import Base: getindex, copy;

export euclidean_distance, mean_square_error, root_mean_square_error,
        mean_error, manhattan_distance, mean_boolean_error, hamming_distance, gaussian,
        DataSet, set_problem, attribute_index, check_dataset_fields,
        check_example, update_values, add_example, remove_examples, sanitize, summarize, copy,
        classes_to_numbers, split_values_by_classes, find_means_and_deviations,
        AbstractCountingProbabilityDistribution,
        CountingProbabilityDistribution, add, smooth_for_observation, getindex, top, sample,
        AbstractLearner, PluralityLearner, predict,
        AbstractNaiveBayesModel, NaiveBayesLearner,
        NaiveBayesDiscreteModel, NaiveBayesContinuousModel, NearestNeighborLearner,
        AbstractDecisionTreeNode, DecisionLeafNode, DecisionForkNode, classify,
        DecisionTreeLearner, plurality_value,
        RandomForest, data_bagging, feature_bagging,
        DecisionListLearner, decision_list_learning,
        NeuralNetworkUnit, NeuralNetworkLearner, neural_network,
        back_propagation_learning, random_weights,
        PerceptronLearner, EnsembleLearner,
        weighted_mode, AdaBoostLearner, adaboost!, weighted_replicate,
        partition_dataset, cross_validation, cross_validation_wrapper,
        RestaurantDataSet, SyntheticRestaurantDataSet,
        MajorityDataSet, ParityDataSet, XorDataSet, ContinuousXorDataSet;

function euclidean_distance(X::AbstractVector, Y::AbstractVector)
    return sqrt(sum(((x - y)^2) for (x, y) in zip(X, Y)));
end

function mean_square_error(X::AbstractVector, Y::AbstractVector)
    return mean(((x - y)^2) for (x, y) in zip(X, Y));
end

function root_mean_square_error(X::AbstractVector, Y::AbstractVector)
    return sqrt(mean_square_error(X, Y));
end

function mean_error(X::AbstractVector, Y::AbstractVector)
    return mean(abs(x - y) for (x, y) in zip(X, Y));
end

function manhattan_distance(X::AbstractVector, Y::AbstractVector)
    return sum(abs(x - y) for (x, y) in zip(X, Y));
end

function mean_boolean_error(X::AbstractVector, Y::AbstractVector)
    return mean((x != y) for (x, y) in zip(X, Y));
end

function hamming_distance(X::AbstractVector, Y::AbstractVector)
    return sum((x != y) for (x, y) in zip(X, Y));
end

"""
    gaussian(mean::Float64, standard_deviation::Float64, x::Number)

Return the probability density of the gaussian distribution for variable 'x' given
the mean 'mean' and standard deviation 'standard_deviation'.
"""
function gaussian(mean::Number, standard_deviation::Number, x::Number)
    return ((Float64(1)/(sqrt(2 * pi) * Float64(standard_deviation))) *
            (e^(-0.5*((Float64(x) - Float64(mean))/Float64(standard_deviation))^2)));
end

#=

    DataSet is a data set used by machine learning algorithms.

=#
type DataSet
    name::String
    source::String
    examples::AbstractMatrix
    attributes::AbstractVector
    attributes_names::AbstractVector
    values::AbstractVector
    exclude::AbstractVector
    distance::Function
    inputs::AbstractVector
    target::Int64

    function DataSet(;name::String="", source::String="", attributes::Union{Void, AbstractVector}=nothing,
                    attributes_names::Union{Void, String, AbstractVector}=nothing,
                    examples::Union{Void, String, AbstractMatrix}=nothing,
                    values::Union{Void, AbstractVector}=nothing,
                    inputs::Union{Void, String, AbstractVector}=nothing, target::Union{Int64, String}=-1,
                    exclude::AbstractVector=[], distance::Function=mean_boolean_error)
        # Use a matrix instead of array of arrays
        local examples_array::AbstractMatrix;
        if (typeof(examples) <: String)
            examples_array = readcsv(examples);
        elseif (typeof(examples) <: Void)
            # 'name'.csv must be in current directory.
            examples_array = readcsv(name*".csv");
        else
            examples_array = examples;
        end
        examples_array = map((function(element)
                                if (typeof(element) <: AbstractString)
                                    return strip(element);
                                else
                                    return element;
                                end
                            end),
                            examples_array);
        local attributes_array::AbstractVector;
        if (typeof(attributes) <: Void)
            attributes_array = collect(1:getindex(size(examples_array), 2));
        else
            attributes_array = attributes;
        end
        local attributes_names_array::AbstractVector;
        if (typeof(attributes_names) <: String)
            attributes_names_array = map(String, split(attributes_names));
        elseif (!(typeof(attributes_names) <: Void) && (length(attributes_names) != 0))
            attributes_names_array = attributes_names;
        else
            attributes_names_array = attributes_array;
        end
        local values_is_set::Bool;
        local new_values::AbstractVector;
        if (typeof(values) <: Void)
            values_is_set = false;
            new_values = [];
        else
            values_is_set = true;
            new_values = values;
        end

        # Construct new DataSet without 'inputs' and 'target' fields.
        local ds::DataSet = new(name, source, examples_array, attributes_array, attributes_names_array,
                                new_values, exclude, distance);

        # Set 'inputs' and 'target' fields of newly constructed DataSet.
        set_problem(ds, target, inputs, exclude);
        check_dataset_fields(ds, values_is_set);

        return ds;
    end

    function DataSet(name::String,
                    source::String,
                    examples::AbstractMatrix,
                    attributes::AbstractVector,
                    attributes_names::AbstractVector,
                    values::AbstractVector,
                    exclude::AbstractVector,
                    distance::Function,
                    inputs::AbstractVector,
                    target::Int64)
        return new(name, source, examples, attributes, attributes_names, values, exclude, distance, inputs, target);
    end
end

function set_problem(ds::DataSet, target::Int64, inputs::Void, exclude::AbstractVector)
    ds.target = attribute_index(ds, target);
    local mapped_exclude::AbstractVector = map(attribute_index, (ds for i in exclude), exclude);
    ds.inputs = collect(a for a in ds.attributes if ((a != ds.target) && (!(a in mapped_exclude))));
    if (length(ds.values) == 0)
        update_values(ds);
    end
    nothing;
end

function set_problem(ds::DataSet, target::Int64, inputs::AbstractVector, exclude::AbstractVector)
    ds.target = attribute_index(ds, target);
    ds.inputs = removeall(inputs, ds.target);
    if (length(ds.values) == 0)
        update_values(ds);
    end
    nothing;
end

function set_problem(ds::DataSet, target::String, inputs::Void, exclude::AbstractVector)
    ds.target = attribute_index(ds, target);
    local mapped_exclude::AbstractVector = map(attribute_index, (ds for i in exclude), exclude);
    ds.inputs = collect(a for a in ds.attributes if ((a != ds.target) && (!(a in mapped_exclude))));
    if (length(ds.values) == 0)
        update_values(ds);
    end
    nothing;
end

function set_problem(ds::DataSet, target::String, inputs::AbstractVector, exclude::AbstractVector)
    ds.target = attribute_index(ds, target);
    ds.inputs = removeall(inputs, ds.target);
    if (length(ds.values) == 0)
        update_values(ds);
    end
    nothing;
end

function attribute_index(ds::DataSet, attribute::String)
    return utils.index(ds.attributes_names, attribute);
end

function attribute_index(ds::DataSet, attribute::Int64)
    # Julia counts from 1.
    if (attribute < 0)
        return length(ds.attributes) + attribute + 1;
    elseif (attribute > 0)
        return attribute;
    else
        error("attribute_index: \"", attribute, "\" is not a valid index for an array!");
    end
end

function check_dataset_fields(ds::DataSet, values_bool::Bool)
    if (length(ds.attributes_names) != length(ds.attributes))
        error("check_dataset_fields(): The lengths of 'attributes_names' and 'attributes' must match!");
    elseif (!(ds.target in ds.attributes))
        error("check_dataset_fields(): The target attribute was not found in 'attributes'!");
    elseif (ds.target in ds.inputs)
        error("check_dataset_fields(): The target attribute should not be in the inputs!");
    elseif (!issubset(Set(ds.inputs), Set(ds.attributes)))
        error("check_dataset_fields(): The 'inputs' field must be a subset of 'attributes'!");
    end
    if (values_bool)
        map(check_example, (ds for i in ds.examples), ds.examples);
    end
    nothing;
end

function check_example(ds::DataSet, example::AbstractVector)
    if (length(ds.values) != 0)
        for attribute in ds.attributes
            if (!(example[attribute] in ds.values[attribute]))
                error("check_example(): Found bad value ", example[attribute], " for attribute ", 
                        ds.attribute_names[attribute], " in ", example, "!");
            end
        end
    end
    nothing;
end

function check_example(ds::DataSet, example::AbstractMatrix)
    if (length(ds.values) != 0)
        for attribute in ds.attributes
            if (!(example[attribute, :] in ds.values[attribute]))
                error("check_example(): Found bad value ", example[attribute, :], " for attribute ", 
                        ds.attribute_names[attribute], " in ", example, "!");
            end
        end
    end
    nothing;
end

function update_values(ds::DataSet)
    ds.values = collect(collect(Set(ds.examples[:, i])) for i in 1:size(ds.examples)[2]);
    nothing;
end

function add_example(ds::DataSet, example::AbstractVector)
    check_example(ds, example);
    vcat(ds.examples, transpose(example));
    nothing;
end

function remove_examples(ds::DataSet)
    local updated_examples::AbstractVector = [];
    for i in size(ds.examples)[1]
        if (!("" in ds.examples[i, :]))
            push!(update_examples, transpose(ds.examples[i, :]));
        end
    end
    ds.examples = reduce(vcat, Array{Any, 2}(), updated_examples);
    update_values(ds);
end

function remove_examples(ds::DataSet, value::String)
    local updated_examples::AbstractVector = [];
    for i in size(ds.examples)[1]
        if (!(value in ds.examples[i, :]))
            push!(update_examples, transpose(ds.examples[i, :]));
        end
    end
    ds.examples = reduce(vcat, Array{Any, 2}(), updated_examples);
    update_values(ds);
end

"""
    sanitize(ds::DataSet, example::AbstractVector)

Return a copy of the given array 'example', such that non-input attributes are removed.
"""
function sanitize(ds::DataSet, example::AbstractVector)
    local sanitized_example::AbstractVector = [];
    for (i, example_item) in enumerate(example)
        if (i in ds.inputs)
            push!(sanitized_example, example_item);
        end
    end
    return sanitized_example;
end

"""
    classes_to_numbers(ds::DataSet, classes::Void)
    classes_to_numbers(ds::DataSet, classes::AbstractVector)

Set the classifications of each example in ds.examples as numbers based on the given 'classes'.
"""
function classes_to_numbers(ds::DataSet, classes::Void)
    local new_classes::AbstractVector = sort(ds.values[ds.target]);
    for i in 1:size(ds.examples)[1]
        local index_val::Int64 = utils.index(new_classes, ds.examples[i, ds.target]);
        if (index_val == -1)
            error("classes_to_numbers(): Could not find ", ds.examples[i, ds.target], " in ", new_classes, "!");
        end
        ds.examples[i, ds.target] = index_val;
    end
    nothing;
end

function classes_to_numbers(ds::DataSet, classes::AbstractVector)
    local new_classes::AbstractVector;
    if (length(classes) == 0)
        new_classes = sort(ds.values[ds.target]);
    else
        new_classes = classes;
    end
    for i in 1:size(ds.examples)[1]
        local index_val::Int64 = utils.index(new_classes, ds.examples[i, ds.target]);
        if (index_val == -1)
            error("classes_to_numbers(): Could not find ", ds.examples[i, ds.target], " in ", new_classes, "!");
        end
        ds.examples[i, ds.target] = index_val;
    end
    nothing;
end

function split_values_by_classes(ds::DataSet)
    local buckets::Dict = Dict();
    local target_names::AbstractVector = ds.values[ds.target];

    for example in (ds.examples[i, :] for i in 1:size(ds.examples)[1])
        local item::AbstractVector = collect(attribute for attribute in example
                                            if (!(attribute in target_names)));
        push!(get!(buckets, example[ds.target], []), item);
    end

    return buckets;
end

function find_means_and_deviations(ds::DataSet)
    local target_names::AbstractVector = ds.values[ds.target];
    local feature_numbers::Int64 = length(ds.inputs);
    local item_buckets::Dict = split_values_by_classes(ds);
    local means::Dict = Dict();
    local deviations::Dict = Dict();
    local key_initial_value::AbstractVector = collect(0.0 for i in 1:feature_numbers);

    for target_name in target_names
        local features = collect(Array{Float64, 1}() for i in 1:feature_numbers);
        for item in item_buckets[target_name]
            for i in 1:feature_numbers
                push!(features[i], item[i]);
            end
        end
        for i in 1:feature_numbers
            get!(means, target_name, copy(key_initial_value))[i] = mean(features[i]);
            get!(deviations, target_name, copy(key_initial_value))[i] = std(features[i]);
        end
    end
    return means, deviations;
end

function summarize(ds::DataSet)
    return @sprintf("<DataSet(%s): %d examples, %d attributes>", ds.name, length(ds.examples), length(ds.attributes));
end

copy(ds::DataSet) = DataSet(identity(ds.name),
                            identity(ds.source),
                            copy(ds.examples),
                            copy(ds.attributes),
                            copy(ds.attributes_names),
                            copy(ds.values),
                            copy(ds.exclude),
                            ds.distance,
                            copy(ds.inputs),
                            identity(ds.target));

abstract AbstractCountingProbabilityDistribution;

#=

    CountingProbabilityDistribution is a probability distribution for counting

    observations. Unlike the other implementations of AbstractProbabilityDistribution,

    CountingProbabilityDistribution calculates the key's probability when

    accessing the CountingProbabilityDistribution by the given key.

=#
type CountingProbabilityDistribution <: AbstractCountingProbabilityDistribution
    dict::Dict
    number_of_observations::Int64
    default::Int64
    sample_function::Nullable{Function}

    function CountingProbabilityDistribution(observations::AbstractVector; default::Int64=0)
        local cpd::CountingProbabilityDistribution = new(Dict(), 0, default, Nullable{Function}());
        for observation in observations
            add(cpd, observation);
        end
        return cpd;
    end

    function CountingProbabilityDistribution(; default::Int64=0)
        local cpd::CountingProbabilityDistribution = new(Dict(), 0, default, Nullable{Function}());
        return cpd;
    end
end

"""
    add{T <: AbstractCountingProbabilityDistribution}(cpd::T, observation)

Add observation 'observation' to the probability distribution 'cpd'.
"""
function add{T <: AbstractCountingProbabilityDistribution}(cpd::T, observation)
    smooth_for_observation(cpd, observation);
    cpd.dict[observation] = cpd.dict[observation] + 1;
    cpd.number_of_observations = cpd.number_of_observations + 1;
    cpd.sample_function = Nullable{Function}();
    nothing;
end

"""
    smooth_for_observation{T <: AbstractCountingProbabilityDistribution}(cpd::T, observation)

Initialize observation 'observation' in the distribution 'cpd' if the observation doesn't
exist in the distribution yet.
"""
function smooth_for_observation{T <: AbstractCountingProbabilityDistribution}(cpd::T, observation)
    if (!(observation in keys(cpd.dict)))
        cpd.dict[observation] = cpd.default;
        cpd.number_of_observations = cpd.number_of_observations + cpd.default;
        cpd.sample_function = Nullable{Function}();
    end
    nothing;
end

"""
    getindex{T <: AbstractCountingProbabilityDistribution}(cpd::T, key)

Return the probability of the given 'key'.
"""
function getindex{T <: AbstractCountingProbabilityDistribution}(cpd::T, key)
    smooth_for_observation(cpd, key);
    return (Float64(cpd.dict[key]) / Float64(cpd.number_of_observations));
end

"""
    top{T <: AbstractCountingProbabilityDistribution}(cpd::T, n::Int64)

Return an array of (observation_count, observation) tuples such that the array
does not exceed length 'n'.
"""
function top{T <: AbstractCountingProbabilityDistribution}(cpd::T, n::Int64)
    local observations::AbstractVector = sort(collect(reverse((i...)) for i in cpd.dict),
                                                lt=(function(p1::Tuple{Number, Any}, p2::Tuple{Number, Any})
                                                        return (p1[1] > p2[1]);
                                                    end));
    if (length(observations) <= n)
        return observations;
    else
        return observations[1:n];
    end
end

"""
    sample(cpd::CountingProbabilityDistribution)

Return a random sample from the probability distribution 'cpd'.
"""
function sample(cpd::CountingProbabilityDistribution)
    if (isnull(cpd.sample_function))
        cpd.sample_function = weighted_sampler(collect(keys(cpd.dict)), collect(values(cpd.dict)));
    end
    return get(cpd.sample_function)();
end

abstract AbstractLearner;

type PluralityLearner{T} <: AbstractLearner
    most_popular::T

    function PluralityLearner{T}(mp::T)
        return new(mp);
    end
end

function PluralityLearner(ds::DataSet)
    most_popular = mode(example[ds.target]
                        for example in (ds.examples[i, :] for i in 1:size(ds.examples)[1]));
    return PluralityLearner{typeof(most_popular)}(most_popular);
end

function predict(pl::PluralityLearner, example::AbstractVector)
    return pl.most_popular;
end

abstract AbstractNaiveBayesModel;

type NaiveBayesLearner <: AbstractLearner
    model::AbstractNaiveBayesModel

    function NaiveBayesLearner(ds::DataSet; continuous::Bool=true)
        if (continuous)
            return new(NaiveBayesContinuousModel(ds));
        else
            return new(NaiveBayesDiscreteModel(ds));
        end
    end
end

function predict(nbl::NaiveBayesLearner, example::AbstractVector)
    return predict(nbl.model, example);
end

type NaiveBayesDiscreteModel <: AbstractNaiveBayesModel
    dataset::DataSet
    target_values::AbstractVector
    target_distribution::CountingProbabilityDistribution
    attributes_distributions::Dict

    function NaiveBayesDiscreteModel(ds::DataSet)
        local nbdm::NaiveBayesDiscreteModel = new(ds,
                                                ds.values[ds.target],
                                                CountingProbabilityDistribution(ds.values[ds.target]));
        nbdm.attributes_distributions = Dict(Pair((val, attribute), CountingProbabilityDistribution(ds.values[attribute]))
                                            for val in nbdm.target_values
                                            for attribute in ds.inputs);
        for example in (ds.examples[i, :] for i in 1:size(ds.examples)[1])
            target_value = example[ds.target];
            add(nbdm.target_distribution, target_value);
            for attribute in ds.inputs
                add(nbdm.attributes_distributions[(target_value, attribute)], example[attribute]);
            end
        end
        return nbdm;
    end
end

function predict(nbdm::NaiveBayesDiscreteModel, example::AbstractVector)
    return argmax(nbdm.target_values,
                    (function(target_value)
                        return (nbdm.target_distribution[target_value] *
                                prod(nbdm.attributes_distributions[(target_value, attribute)][example[attribute]]
                                    for attribute in nbdm.dataset.inputs));
                    end));
end

type NaiveBayesContinuousModel <: AbstractNaiveBayesModel
    dataset::DataSet
    target_values::AbstractVector
    target_distribution::CountingProbabilityDistribution
    means::Dict
    deviations::Dict

    function NaiveBayesContinuousModel(ds::DataSet)
        local nbcm::NaiveBayesContinuousModel = new(ds,
                                                    ds.values[ds.target],
                                                    CountingProbabilityDistribution(ds.values[ds.target]));
        nbcm.means, nbcm.deviations = find_means_and_deviations(ds);
        return nbcm;
    end
end

function predict(nbcm::NaiveBayesContinuousModel, example::AbstractVector)
    return argmax(nbcm.target_values,
                    (function(target_value)
                        local p::Float64 = nbcm.target_distribution[target_value];
                        for attribute in nbcm.dataset.inputs
                            p = p * gaussian(nbcm.means[target_value][attribute],
                                            nbcm.deviations[target_value][attribute],
                                            example[attribute]);
                        end
                        return p;
                    end));
end

#=

    NearestNeighborLearner uses the k-nearest neighbors lookup for predictions.

=#
type NearestNeighborLearner <: AbstractLearner
    dataset::DataSet
    k::Int64

    function NearestNeighborLearner(ds::DataSet)
        return new(ds, 1);
    end

    function NearestNeighborLearner(ds::DataSet, k::Int64)
        return new(ds, k);
    end
end

function nearest_neighbor_predict_isless(t1::Tuple, t2::Tuple)
    return (t1[1] < t2[1]);
end

function predict(nnl::NearestNeighborLearner, example::AbstractVector)
    local best_distances::AbstractVector = sort(collect((nnl.dataset.distance(dataset_example, example), dataset_example)
                                                        for dataset_example in (nnl.dataset.examples[i, :] for i in 1:size(nnl.dataset.examples)[1])),
                                                lt=nearest_neighbor_predict_isless);
    if (length(best_distances) > nnl.k)
        best_distances = best_distances[1:nnl.k];
    end
    return mode(dataset_example[nnl.dataset.target] for (distance, dataset_example) in best_distances);
end

abstract AbstractDecisionTreeNode;

type DecisionLeafNode{T} <: AbstractDecisionTreeNode
    result::T

    function DecisionLeafNode{T}(result::T)
        return new(result);
    end
end

function classify(dl::DecisionLeafNode, example::AbstractVector)
    return dl.result;
end

DecisionLeafNode(result) = DecisionLeafNode{typeof(result)}(result);

type DecisionForkNode <: AbstractDecisionTreeNode
    attribute::Int64
    attribute_name::Nullable
    default_child::Nullable{DecisionLeafNode}
    branches::Dict

    function DecisionForkNode(attribute::Int64;
                        attribute_name::Union{Int64, String, Void}=nothing,
                        default_child::Union{DecisionLeafNode, Void}=nothing,
                        branches::Union{Dict, Void}=nothing)
        local new_attribute_name::Nullable;
        local new_branches::Dict;
        if (typeof(attribute_name) <: Void)
            new_attribute_name = Nullable(attribute);
        else
            new_attribute_name = Nullable(attribute_name);
        end
        if (typeof(branches) <: Void)
            new_branches = Dict();
        else
            new_branches = branches;
        end
        return new(attribute, new_attribute_name, Nullable{DecisionLeafNode}(default_child), new_branches);
    end
end

function classify(df::DecisionForkNode, example::AbstractVector)
    attribute_value = example[df.attribute];
    if (haskey(df.branches, attribute_value))
        return classify(df.branches[attribute_value], example);
    else
        return classify(get(df.default_child), example);
    end
end

function add(dfn::DecisionForkNode, key::Real, subtree)
    dfn.branches[key] = subtree;
    nothing;
end

function summarize(dfn::DecisionForkNode)
    return @sprintf("DecisionForkNode(%s, %s, %s)", repr(dfn.attribute), repr(dfn.attribute_name), repr(dfn.branches));
end

"""
    information_content(values::AbstractVector)

Return the number of bits that represent the probability distribution of non-zero values in 'values'.
"""
function information_content(values::AbstractVector)
    local probabilities::Array{Float64, 1} = normalize(removeall(values, 0), 1);
    if (length(probabilities) == 0)
        return Float64(0);
    else
        return sum((-p * log2(p)) for p in probabilities);
    end
end

function information_gain_content(dataset::DataSet, examples::AbstractMatrix)
    return information_content(collect(count((function(example)
                                            return (example[dataset.target] == value);
                                        end), (examples[i,:] for i in 1:size(examples)[1]))
                                    for value in dataset.values[dataset.target]));
end

"""
    matrix_vcat(args::Vararg)

Returns an empty matrix when vcat() returns an empty vector, otherwise return vcat(args).
"""
function matrix_vcat(args::Vararg)
    if (length(args) == 0)
        return Array{Any, 2}();
    else
        return vcat(args...);
    end
end

"""
    filter_examples_by_attribute(dataset::DataSet, attribute::Int64, examples::AbstractMatrix)

Return a Base.Generator of (value_i, examples_i) tuples for each value of 'attribute'.
"""
function filter_examples_by_attribute(dataset::DataSet, attribute::Int64, examples::AbstractMatrix)
    return ((value, matrix_vcat((reshape(ex_i, (1, length(ex_i)))
                                                    for ex_i in (examples[i,:] for i in 1:size(examples)[1])
                                                    if (ex_i[attribute] == value))...))
            for value in dataset.values[attribute]);
end

"""
    information_gain(dataset::DataSet, attribute::Int64, examples::AbstractMatrix)

Return the expected reduction in entropy from testing the attribute 'attribute' given
the dataset 'dataset' and matrix 'examples'.
"""
function information_gain(dataset::DataSet, attribute::Int64, examples::AbstractMatrix)
    local N::Float64 = Float64(size(examples)[1]);
    local remainder::Float64 = Float64(sum(((size(examples_i)[1]/N)
                                            * information_gain_content(dataset, examples_i)
                                            for (value, examples_i) in filter_examples_by_attribute(dataset, attribute, examples))));
    return (Float64(information_gain_content(dataset, examples)) - remainder);
end

"""
    plurality_value(dataset::DataSet, examples::AbstractMatrix)

Return a DecisionLeafNode with the result field set to the most common output value
in the given matrix 'examples' (using argmax_random_tie()). 
"""
function plurality_value(dataset::DataSet, examples::AbstractMatrix)
    return DecisionLeafNode(argmax_random_tie(dataset.values[dataset.target],
                                            (function(value)
                                                return count((function(example::AbstractVector)
                                                                return (example[dataset.target] == value);
                                                            end), (examples[i,:] for i in 1:size(examples)[1]));
                                            end)));
end

"""
    decision_tree_learning(dataset::DataSet, examples::AbstractMatrix, attributes::AbstractVector; parent_examples::AbstractMatrix=Array{Any, 2}())

Return a decision tree as a DecisionLeafNode or a DecisionForkNode by applying the decision-tree
learning algorithm (Fig. 18.5) on the given dataset 'dataset', example matrix 'example', attributes
vector 'attributes', and parent examples 'parent_examples'.
"""
function decision_tree_learning(dataset::DataSet, examples::AbstractMatrix, attributes::AbstractVector; parent_examples::AbstractMatrix=Array{Any, 2}())
    # examples is empty
    if (size(examples)[1] == 0)
        return plurality_value(dataset, parent_examples);
    # examples have the same classification
    elseif (all((example[dataset.target] == examples[1, dataset.target])
                for example in (examples[i, :] for i in 1:size(examples)[1])))
        return DecisionLeafNode(examples[1, dataset.target]);
    # attributes is empty
    elseif (length(attributes) == 0)
        return plurality_value(dataset, parent_examples);
    else
        local A::Int64 = argmax_random_tie(attributes,
                                            (function(attribute)
                                                return information_gain(dataset, attribute, examples);
                                            end));
        local tree::DecisionForkNode = DecisionForkNode(A,
                                                    attribute_name=dataset.attributes_names[A],
                                                    default_child=plurality_value(dataset, examples));
        for (v_k, exs) in filter_examples_by_attribute(dataset, A, examples)
            local subtree::AbstractDecisionTreeNode = decision_tree_learning(dataset, exs, removeall(attributes, A), parent_examples=examples);
            add(tree, v_k, subtree);
        end
        return tree;
    end
end

type DecisionTreeLearner <: AbstractLearner
    decision_tree::AbstractDecisionTreeNode

    function DecisionTreeLearner(dataset::DataSet)
        return new(decision_tree_learning(dataset, dataset.examples, dataset.inputs));
    end

    # The following constructor method for DecisionTreeLearner is to be used with
    # cross_validation() for a decision tree constructed in a breadth-first fashion.
    #
    # The breadth-first decision_tree_learning() method is not implemented yet.
    function DecisionTreeLearner(dataset::DataSet, node_count::Int64)
        return new(decision_tree_learning(dataset, dataset.examples, dataset.inputs, node_count));
    end
end

function predict(dtl::DecisionTreeLearner, example::AbstractVector)
    return classify(dtl.decision_tree, example);
end

function data_bagging(dataset::DataSet)
    local n::Int64 = size(dataset.examples)[1];
    local sampled_examples::AbstractVector = weighted_sample_with_replacement(collect(dataset.examples[i, :]
                                                                                    for i in 1:size(dataset.examples)[1]), ones(n), n);
    return reduce(vcat, (reshape(sample_example, (1, length(sample_example)))
                        for sample_example in sampled_examples));
end

function data_bagging(dataset::DataSet, m::Int64)
    local n::Int64 = size(dataset.examples)[1];
    local sampled_examples::AbstractVector = weighted_sample_with_replacement(collect(dataset.examples[i, :]
                                                                                    for i in 1:size(dataset.examples)[1]), ones(n), m);
    return reduce(vcat, (reshape(sample_example, (1, length(sample_example)))
                        for sample_example in sampled_examples));
end

function feature_bagging(dataset::DataSet; p::Float64=0.7)
    local inputs::AbstractVector = collect(i for i in dataset.inputs if (rand(RandomDeviceInstance) < p));
    if (length(inputs) == 0)
        return dataset.inputs;
    else
        return inputs;
    end
end

type RandomForest <: AbstractLearner
    predictors::Array{DecisionTreeLearner, 1}

    function RandomForest(dataset::DataSet; n::Int64=5)
        local predictors::Array{DecisionTreeLearner, 1} = collect(DecisionTreeLearner(DataSet(examples=data_bagging(dataset),
                                                                                                    attributes=dataset.attributes,
                                                                                                    attributes_names=dataset.attributes_names,
                                                                                                    target=dataset.target,
                                                                                                    inputs=feature_bagging(dataset)))
                                                                        for i in 1:n);
        return new(predictors);
    end
end

function predict(rf::RandomForest, example::AbstractVector)
    return mode(predict(predictor, example) for predictor in rf.predictors);
end

function find_test_outcomes_from_examples(ds::DataSet, examples::Set)
    println("find_test_outcomes_from_examples() is not yet implemented!");
    nothing;
end

"""
    decision_list_learning(ds::DataSet, examples::Set)

Return an array of (test::Function, outcome) tuples by using the decision list learning
algorithm (Fig. 18.11) on the given dataset 'ds' and a set of examples 'examples'.
"""
function decision_list_learning(ds::DataSet, examples::Set)
    if (length(examples) == 0)
        return [((function(examples::AbstractVector)
                        return true;
                    end), false)];
    end
    local t::Function;
    local examples_t::Set;
    t, output, examples_t = find_test_outcomes_from_examples(ds, examples);
    if (typeof(t) <: Void)
        error("decision_list_learning(): Could not find valid test 't'!");
    end
    return append!([(t, output)], decision_list_learning(ds, setdiff(examples, examples_t)));
end

type DecisionListLearner <: AbstractLearner
    decision_list::AbstractVector

    function DecisionListLearner(dataset::DataSet)
        return new(decision_list_learning(dataset, Set(dataset.examples[i, :]
                                                        for i in 1:size(dataset.examples)[1])));
    end
end

function predict(dll::DecisionListLearner, examples::AbstractVector)
    for (test, outcome) in dll.decision_list
        if (test(examples))
            return outcome;
        end
    end
    error("predict(): All tests in the generated decision list failed for ", examples, "!");
end

#=

    NeuralNetworkUnit is an unit (node) in a multilayer neural network.

=#
type NeuralNetworkUnit
    weights::AbstractVector
    inputs::AbstractVector
    value::Nullable
    activation::Function

    function NeuralNetworkUnit()
        return new([], [], Nullable(nothing), sigmoid);
    end

    function NeuralNetworkUnit(weights::AbstractVector, inputs::AbstractVector)
        return new(weights, inputs, Nullable(nothing), sigmoid);
    end
end

"""
    neural_network(input_units::Int64, hidden_layers_sizes::Array{Int64, 1}, output_units::Int64)

Return an untrained neural network by using the given the number of input units 'input_units', the
hidden layers' sizes (the hidden layers should not include the input and output layers) in
'hidden_layers_sizes', and the number of output units 'output_units'.
"""
function neural_network(input_units::Int64, hidden_layers_sizes::Array{Int64, 1}, output_units::Int64)
    local layers_sizes::AbstractVector = Array{Int64, 1}();
    if (length(hidden_layers_sizes) == 0)
        push!(layers_sizes, input_units);
        push!(layers_sizes, output_units);
    else
        push!(layers_sizes, input_units);
        append!(layers_sizes, hidden_layers_sizes);
        push!(layers_sizes, output_units);
    end
    local network::AbstractVector = collect(collect(NeuralNetworkUnit()
                                                    for node in 1:layer_size)
                                            for layer_size in layers_sizes);
    for i in 2:length(network)
        for n in network[i]
            for k in network[i - 1]
                push!(n.inputs, k);
                push!(n.weights, 0.0);
            end
        end
    end
    return network;
end

function back_propagation_initialize_examples(examples::AbstractMatrix, idx_i::AbstractVector, idx_t::Int64, o_units::Int64)
    local inputs::Dict = Dict();
    local targets::Dict = Dict();

    for i in 1:size(examples)[1]
        local example::AbstractVector = examples[i, :];
        inputs[i] = collect(example[idx] for idx in idx_i);
        if (o_units > 1)
            local t::AbstractVector = collect(0.0 for j in 1:o_units);
            t[example[idx_t]] = 1.0;
            targets[i] = t;
        else
            targets[i] = [example[idx_t]];
        end
    end
    return inputs, targets;
end

"""
    random_weights(bound_1::Real, bound_2::Real, num_weights::Int64)

Return an array of 'num_weights' weights that are randomly generated with the
bounds 'bound_1' and 'bound_2'.
"""
function random_weights(bound_1::Real, bound_2::Real, num_weights::Int64)
    local minimum_value::Real = min(bound_1, bound_2);
    return collect((minimum_value + (rand(RandomDeviceInstance) * abs(bound_1 - bound_2)))
                    for i in 1:num_weights);
end

"""
    back_propagation_learning!(dataset::DataSet, network::AbstractVector, learning_rate::Float64, epochs::Int64)

Return the trained neural network by applying the back-propagation algorithm (Fig. 18.24) on the
given multilayer neural network 'network', dataset 'dataset', learning rate 'learning_rate', and
the number of epochs 'epochs'.
"""
function back_propagation_learning!(dataset::DataSet, network::AbstractVector, learning_rate::Float64, epochs::Int64)
    for layer in network
        for unit in layer
            unit.weights = random_weights(-0.5, 0.5, length(unit.weights));
        end
    end
    local output_units::AbstractVector = network[end];
    local input_units::AbstractVector = network[1];
    local num_output_units::Int64 = length(output_units);
    local inputs::Dict;
    local targets::Dict;
    local layer::AbstractVector;

    inputs, targets = back_propagation_initialize_examples(dataset.examples, dataset.inputs, dataset.target, num_output_units);

    for epoch in 1:epochs
        for x in 1:size(dataset.examples)[1]
            local input_value::AbstractVector = inputs[x];
            local target_value::AbstractVector = targets[x];

            # Activate the input layer.
            for (v, u) in zip(input_value, input_units)
                u.value = Nullable(v);
            end

            # Propagate the inputs forward.
            for layer in network[2:end]
                for unit in layer
                    local in_j::Real = dot(collect(get(unit_input.value)
                                            for unit_input in unit.inputs),
                                            unit.weights);
                    unit.value = Nullable(unit.activation(in_j));
                end
            end

            # Initialize deltas array with empty vectors.
            local delta::AbstractVector = collect([] for i in 1:length(network));

            # Compute the errors of the mean squared error function.
            local errors::AbstractVector = collect((target_value[i] - get(output_units[i].value))
                                                for i in 1:num_output_units);
            delta[end] = collect((sigmoid_derivative(get(output_units[i].value)) * errors[i])
                                for i in 1:num_output_units);

            # Propagate the deltas backward from ouput layer to input layer.
            local num_hidden_layers::Int64 = length(network) - 2;
            for i in reverse(2:(num_hidden_layers + 1))
                layer = network[i];
                local num_hidden_units::Int64 = length(layer);
                local next_layer::AbstractVector = network[i + 1];
                w = collect(collect(unit.weights[j]
                                    for unit in next_layer)
                            for j in 1:num_hidden_units);

                delta[i] = collect((sigmoid_derivative(get(layer[j].value)) * dot(w[j], delta[i + 1]))
                                    for j in 1:num_hidden_units);
            end

            # Update every weight in network by using the deltas.
            for i in 2:length(network)
                layer = network[i];
                local previous_layer_values::AbstractVector = collect(get(unit.value) for unit in network[i - 1]);
                local num_units::Int64 = length(layer);
                for j in 1:num_units
                    layer[j].weights = (layer[j].weights + ((learning_rate * delta[i][j]) * previous_layer_values));
                end
            end
        end
    end
    return network;
end

#=

    NeuralNetworkLearner contains a multilayer feed-forward neural network that is

    trained by the back-propagation algorithm with the dataset 'dataset', learning rate

    'learning_rate', and number of epochs (our criterion for stopping training) 'epochs'.

=#
type NeuralNetworkLearner <: AbstractLearner
    network::AbstractVector

    function NeuralNetworkLearner(dataset::DataSet;
                                hidden_layers_sizes::AbstractVector=[3],
                                learning_rate::Float64=0.01,
                                epochs::Int64=100)
        local num_input_units::Int64 = length(dataset.inputs);
        local num_output_units::Int64 = length(dataset.values[dataset.target]);
        local nnl::NeuralNetworkLearner = new(neural_network(num_input_units, hidden_layers_sizes, num_output_units));
        nnl.network = back_propagation_learning!(dataset, nnl.network, learning_rate, epochs);
        return nnl;
    end
end

"""
    predict(nnl::NeuralNetworkLearner, example::AbstractVector)

Return a prediction for the given 'example' by using the logistic regression hypothesis on the
values of 'example' and the weights of each respective unit in the neural network 'nnl.network'.
"""
function predict(nnl::NeuralNetworkLearner, example::AbstractVector)
    local input_units::AbstractVector = nnl.network[1];

    # Set the values of the input units to the values of example.
    for (v, u) in zip(example, input_units)
        u.value = Nullable(v);
    end

    # Propagate the example values forward.
    for layer in nnl.network[2:end]
        for unit in layer
            local in_j::Real = dot(collect(get(unit_input.value)
                                    for unit_input in unit.inputs),
                                    unit.weights);
            unit.value = Nullable(unit.activation(in_j));
        end
    end

    local prediction::Int64 = utils.index(nnl.network[end], argmax(nnl.network[end],
                                                                    (function(unit::NeuralNetworkUnit)
                                                                        return get(unit.value);
                                                                    end)));
    if (prediction < 0)
        error("predict(): NeuralNetworkLearner returned invalid array index '", prediction, "'!");
    else
        return prediction;
    end
end

#=

    PerceptronLearner contains a neural network where hidden layers are not used.

    The neural network is trained by the back-propagation algorithm with the given

    dataset 'dataset'.

=#
type PerceptronLearner <: AbstractLearner
    network::AbstractVector

    function PerceptronLearner(dataset::DataSet;
                                learning_rate::Float64=0.01,
                                epochs::Int64=100)
        local num_input_units::Int64 = length(dataset.inputs);
        local num_output_units::Int64 = length(dataset.values[dataset.target]);
        local hidden_layers_sizes::AbstractVector = Array{Int64, 1}();
        local pl::PerceptronLearner = new(neural_network(num_input_units, hidden_layers_sizes, num_output_units));
        pl.network = back_propagation_learning!(dataset, pl.network, learning_rate, epochs);
        return pl;
    end
end

"""
    predict(pl::PerceptronLearner, example::AbstractVector)

Return a prediction for the given 'example' by using the logistic regression hypothesis on the
values of 'example' and the weights of each respective unit in the neural network 'pl.network'.
"""
function predict(pl::PerceptronLearner, example::AbstractVector)
    local output_units::AbstractVector = pl.network[end];

    # Propagate the example values forward.
    for unit in output_units
        unit.value = Nullable(unit.activation(dot(example, unit.weights)));
    end

    local prediction::Int64 = utils.index(pl.network[end], argmax(pl.network[end],
                                                                (function(unit::NeuralNetworkUnit)
                                                                    return get(unit.value);
                                                                end)));
    if (prediction < 0)
        error("predict(): PerceptronLearner returned invalid array index '", prediction, "'!");
    else
        return prediction;
    end
end

#=

    LinearRegressionLearner creates a linear model by applying multivariate

    linear regression to the given dataset. The regressands are assumed to be

    real numbers.

=#
type LinearRegressionLearner <: AbstractLearner
    dataset::DataSet
    weights::AbstractVector

    function LinearRegressionLearner(dataset::DataSet;
                                    learning_rate::Float64=0.01,
                                    epochs::Int64=100)
        # Initialize random weights
        local lrl::LinearRegressionLearner = new(dataset, random_weights(-0.5, 0.5, length(dataset.inputs) + 1));

        # Initialize an array of x_i (vectors) such that the first component of each vector is 1.0.
        local X::AbstractVector = collect(vcat(1.0, dataset.values[i]) for i in dataset.inputs);

        for i in 1:epochs
            local error::AbstractVector = Array{Float64, 1}();
            for example in (dataset.examples[i, :] for i in 1:size(dataset.examples)[1])
                # The target attribute will be ignored in the evaluation of h_w(x).
                local x::AbstractVector = vcat(1, sanitize(dataset, example));
                local h_x::Float64 = dot(lrl.weights, x);
                # The error component is the square of the difference between the regressand y and h_w(x).
                local y::Float64 = examples[dataset.target];
                push!(error, t - y);
            end

            # Use the least mean squares algorithm for stochastic gradient descent.
            for i in 1:length(lrl.weights)
                lrl.weights[i] = lrl.weights[i] + ((learning_rate * dot(error, X))/size(dataset.examples)[1]);
            end
        end

        return lrl;
    end
end

function predict(lrl::LinearRegressionLearner, example::AbstractVector)
    return dot(lrl.weights, vcat(1.0, sanitize(lrl.dataset, example)));
end

#=

    EnsembleLearner is a learner that uses an ensemble of hypotheses generated by the given

    learner constructors to vote on the best classification when predicting.

=#
type EnsembleLearner <: AbstractLearner
    predictors::AbstractVector

    function EnsembleLearner(dataset::DataSet, learners::AbstractVector)
        if (!(all(((typeof(learner) <: DataType) && (learner <: AbstractLearner))
                    for learner in learners)))
            error("EnsembleLearner(): All items in learners must be a subtype of AbstractLearner!");
        end
        return new(collect(learner(dataset) for learner in learners));
    end
end

function predict(el::EnsembleLearner, example::AbstractVector)
    return mode(predict(predictor, example) for predictor in predictors);
end

"""
    weighted_mode(values::String, weights::AbstractVector)
    weighted_mode(values::AbstractVector, weights::AbstractVector)

Return the value from 'values' with the largest cumulative weight.
"""
function weighted_mode(values::String, weights::AbstractVector)
    local values_array::AbstractVector = map(String, (collect(char) for char in map(Char, values.data)));
    local weight_values::Dict = Dict();
    for (value, weight) in zip(values_array, weights)
        if (haskey(weight_values, value))
            weight_values[value] = weight_values[value] + weight;
        else
            weight_values[value] = weight;
        end
    end
    return reduce((function(t1::Pair, t2::Pair)
                        if (t1[2] < t2[2])
                            return t2;
                        else
                            return t1;
                        end
                    end),
                    weight_values)[1];
end

function weighted_mode(values::AbstractVector, weights::AbstractVector)
    local weight_values::Dict = Dict();
    for (value, weight) in zip(values, weights)
        if (haskey(weight_values, value))
            weight_values[value] = weight_values[value] + weight;
        else
            weight_values[value] = weight;
        end
    end
    return reduce((function(t1::Pair, t2::Pair)
                        if (t1[2] < t2[2])
                            return t2;
                        else
                            return t1;
                        end
                    end),
                    weight_values)[1];
end

"""
    weighted_majority(predictors::AbstractVector, weights::AbstractVector)

Return a function that returns the highest weighted vote based on the
weights and predict() methods for the given learners 'predictors'.
"""
function weighted_majority(predictors::AbstractVector, weights::AbstractVector)
    return (function(examples::AbstractVector)
                return weighted_mode(collect(predict(predictor, example)
                                            for predictor in predictors),
                                    weights);
            end);
end

"""
    weighted_replicate(seq::AbstractVector, weights::AbstractVector, n::Int64)
    weighted_replicate(seq::AbstractMatrix, weights::AbstractVector, n::Int64)

Return an array of 'n' examples where each example is proportional to the corresponding
weight in the given weights array 'weights'. These 'n' examples are sampled from the given
array 'seq' by using weighted_sample_with_replacement().
"""
function weighted_replicate(seq::AbstractVector, weights::AbstractVector, n::Int64)
    if (length(seq) != length(weights))
        error("weighted_replicate(): The length of 'seq' and 'weights' must match!");
    end

    local normalized_weights::Array{Float64, 1} = normalize(weights, 1);
    local integer_multiples::Array{Int64, 1} = collect(Int64(floor(weight * n))
                                                        for weight in normalized_weights);
    local fractions::Array{Float64, 1} = collect(((weight * n) % 1) for weight in normalized_weights);

    return append!(reduce(vcat, [], collect(fill(x, num_x)
                                            for (x, num_x) in zip(seq, integer_multiples))),
                    weighted_sample_with_replacement(seq, fractions, (n - sum(integer_multiples))));
end

function weighted_replicate(seq::AbstractMatrix, weights::AbstractVector, n::Int64)
    if (size(seq)[1] != length(weights))
        error("weighted_replicate(): The length of 'seq' and 'weights' must match!");
    end

    local normalized_weights::Array{Float64, 1} = normalize(weights, 1);
    local integer_multiples::Array{Int64, 1} = collect(Int64(floor(weight * n))
                                                        for weight in normalized_weights);
    local fractions::Array{Float64, 1} = collect(((weight * n) % 1) for weight in normalized_weights);

    local integer_weighted::AbstractMatrix = vcat(collect(vcat((reshape(x, 1, length(x))
                                                                for i in 1:num_x)...)
                                                for (x, num_x) in zip((seq[i, :]
                                                                        for i in 1:size(seq)[1]),
                                                                        integer_multiples)
                                                if (num_x > 0))...);
    if ((n - sum(integer_multiples)) > 0)
        local fraction_weighted::AbstractMatrix = vcat(collect(reshape(sample, 1, length(sample))
                                                                for sample in weighted_sample_with_replacement(
                                                                    collect(seq[i, :] for i in 1:size(seq)[1]),
                                                                    fractions,
                                                                    (n - sum(integer_multiples))))...);

        return vcat(integer_weighted, fraction_weighted);
    else
        return integer_weighted;
    end
end

function reweighted_dataset(dataset::DataSet, weights::AbstractVector, n::Int64)
    local dataset_copy::DataSet = copy(dataset);
    dataset_copy.examples = weighted_replicate(dataset_copy.examples, weights, n);
    return dataset_copy;
end

function reweighted_dataset(dataset::DataSet, weights::AbstractVector)
    local n::Int64 = size(dataset.examples)[1];
    local dataset_copy::DataSet = copy(dataset);
    dataset_copy.examples = weighted_replicate(dataset_copy.examples, weights, n);
    return dataset_copy;
end


#=

    AdaBoostLearner contains the weights and hypotheses generated by adaboost!().

=#
type AdaBoostLearner <: AbstractLearner
    h::AbstractVector
    z::AbstractVector
    hypothesis::Function

    function AdaBoostLearner(dataset::DataSet, L::DataType, K::Int64)
        if (!(L <: AbstractLearner))
            error("AdaBoostLearner(): The learner ", L, " is not a subtype of AbstractLearner!");
        end
        local abl::AdaBoostLearner = new(Array{Float64, 1}(), Array{Float64, 1}());
        abl.hypothesis = adaboost!(abl, dataset, L, K);
        return abl;
    end
end

function predict(abl::AdaBoostLearner, example::AbstractVector)
    return abl.hypothesis(example);
end

"""
    adaboost!(abl::AdaBoostLearner, dataset::DataSet, L::DataType, K::Int64)

Return a weighted-majority hypothesis by using the AdaBoost algorithm (Fig. 18.34)
on the given dataset 'dataset', learning algorithm 'L', and number of hypotheses to
use in ensemble learning 'K'.

This function sets abl.h to the vector of 'K' hypotheses and abl.z to the vector
of 'K' hypothesis weights.
"""
function adaboost!(abl::AdaBoostLearner, dataset::DataSet, L::DataType, K::Int64)
    local w::AbstractVector = fill((1/size(dataset.examples)[1]), size(dataset.examples)[1]);
    for k in 1:K
        local h_k::AbstractLearner = L(reweighted_dataset(dataset, w));
        push!(abl.h, h_k);
        local error::Float64 = sum(weight
                                    for (example, weight) in zip((dataset.examples[i, :]
                                                                for i in 1:size(dataset.examples)[1]),
                                                                w));
        for (j, example) in enumerate(dataset.examples[i, :] for i in 1:size(dataset.examples)[1])
            if (example[dataset.target] == predict(h_k, example))
                w[j] = w[j] * error;
            end
        end
        w = normalize(w, 1);
        push!(abl.z, log((1.0 - error)/error));
    end
    return weighted_majority(abl.h, abl.k);
end

"""
    grade_learner(learner::AbstractLearner, tests::AbstractVector)

Return a score for the given learner 'learner' and the tests 'tests' (an array of (example, output)).
"""
function grade_learner(learner::AbstractLearner, tests::AbstractVector)
    return mean(Float64(predict(learner, X) == y) for (X, y) in tests);
end

"""
    error_ratio(learner::AbstractLearner, dataset::DataSet, examples::Union{Void, AbstractMatrix}=nothing, verbose::Int64=0)

Return the proportion of examples that were not correctly predicted.

If 'verbose' is set to 0, this function will not print extra messages.
If 'verbose' is set to 1, this function will print messages when a prediction fails.
If 'verbose' is set to 2 or more, this function will print messages for each prediction made.
"""
function error_ratio(learner::AbstractLearner, dataset::DataSet, examples::Union{Void, AbstractMatrix}=nothing, verbose::Int64=0)
    local new_examples::AbstractMatrix;
    if (typeof(examples) <: Void)
        new_examples = dataset.examples;
    else
        new_examples = examples;
    end
    if (size(new_examples)[1] == 0)
        return 0.0;
    end
    local correct::Float64 = 0.0;
    for example in (new_examples[i, :] for i in 1:size(new_examples)[1])
        desired = example[dataset.target];
        output = predict(learner, sanitize(dataset, example));
        if (output == desired)
            correct = correct + 1;
            if (verbose >= 2)
                println("   OK:  Got ", desired, " for ", example, "!");
            end
        elseif (verbose > 0)
            println("WRONG: Got ", output, ", expected ", desired, " for ", example, "!");
        end
    end
    return 1.0 - (correct / size(new_examples)[1]);
end

"""
    partition_dataset(dataset::DataSet, start_index::Real, end_index::Real)

Partition the examples of the given dataset 'dataset' into a set of examples for training
and a separate set of examples for validation.
"""
function partition_dataset(dataset::DataSet, start_index::Real, end_index::Real)
    local int_start::Int64 = Int64(floor(start_index));
    local int_end::Int64 = Int64(floor(end_index));
    local training_set::AbstractMatrix = vcat(dataset.examples[1:(int_start - 1), :],
                                                dataset.examples[int_end:end, :]);
    local validation_set::AbstractMatrix = dataset.examples[int_start:(int_end - 1), :];
    return training_set, validation_set;
end

function cross_validation(learner::DataType, size::Int64, k::Int64, dataset::DataSet, trials::Int64)
    local error_T::AbstractVector = [];
    local error_V::AbstractVector = [];
    for i in 1:trials
        new_error_T, new_error_V = cross_validation(learner, size, k, dataset);
        push!(error_T, new_error_T);
        push!(error_V, new_error_V);
    end
    return mean(error_T), mean(error_V);
end

function cross_validation(learner::DataType, size::Int64, k::Int64, dataset::DataSet)
    local fold_errT::Float64 = 0.0;
    local fold_errV::Float64 = 0.0;
    local num_examples::Int64 = size(dataset.examples)[1];
    for fold in 1:k
        local original_set::AbstractMatrix = dataset.examples;
        local training_set::AbstractMatrix;
        local validation_set::AbstractMatrix;
        training_set, validation_set = partition_dataset(dataset,
                                                        ((fold * n)/k),
                                                        (((fold * n) + n)/k));
        dataset.examples = training_set;

        # Section 18.4.1 (Model selection: Complexity vs goodness of fit) suggests using
        # 'size' as the number of nodes to use in a breadth-first decision tree created
        # by a decision-tree learning algorithm.
        #
        # Currently, decision_tree_learning() traverses the decision tree depth-first.
        local h::AbstractLearner = learner(dataset, size);
        fold_errT = fold_errT + error_ratio(h, dataset, training_set);
        fold_errV = fold_errV + error_ratio(h, dataset, validation_set);
        dataset.examples = original_set;
    end
    return (fold_errT/k), (fold_errV/k);
end

"""
    cross_validation_wrapper(learner::DataType, dataset::DataSet; trials::Int64=1, k::Int64=10)

Apply the cross-validation algorithm (Fig. 18.8) on the given learning algorithm 'learner',
the number of equal subsets to make from splitting the dataset 'k', the dataset 'dataset',
and the number of trials to make for a specific size 'trials'.

Return a trained learner if the errors from the training sets converge, otherwise raise an error
for reaching the max Int64 value before integer overflow.
"""
function cross_validation_wrapper(learner::DataType, dataset::DataSet; trials::Int64=1, k::Int64=10)
    local error_training::Array{Float64, 1} = Array{Float64}();
    local error_validation::Array{Float64, 1} = Array{Float64}();
    local size::Int64 = 1;
    while (true)
        local errT::Float64;
        local errV::Float64;

        errT, errV = cross_validation(learner, size, k, dataset, trials);

        if (length(error_training) != 0)
            if (abs(error_training[end] - errT) <= (0.000001 * max(abs(error_training[end]), abs(errT))))
                local best_size::Int64 = -1;
                local minimum_error::Float64 = Inf64;
                for i in 1:size
                    if (error_validation[i] < minimum_error)
                        minimum_error = error_validation[i];
                        best_size = i;
                    end
                end
                return learner(dataset, best_size);
            end
        end

        if (size == typemax(Int64))
            error("cross_validation(): The 'size' variable will integer overflow!");
        end
        size = size + 1;
    end
end

"""
    leave_one_out_cross_validation(learner::DataType, dataset::DataSet)

Apply the leave-one-out cross-validation algorithm (LOOCV) on the given learning algorithm 'learner'
and the dataset 'dataset'. The number of equal subsets to make from splitting the dataset, 'k', is
set to the number of examples in the dataset.
"""
function leave_one_out_cross_validation(learner::DataType, dataset::DataSet)
    return cross_validation_wrapper(learner, dataset, k=size(dataset.examples)[1]);
end

"""
    learning_curve(learner::DataType, dataset::DataSet; trials::Int64=10, sizes::AbstractVector=[])

Return an array of (size, score) tuples which represent the data points of our learning curve.
"""
function learning_curve(learner::DataType, dataset::DataSet; trials::Int64=10, sizes::AbstractVector=[])
    if (length(sizes) == 0)
        sizes = collect(2:2:(size(dataset.examples)[1] - 2));
    end

    local original_set::AbstractMatrix = dataset.examples;

    # Randomly shuffle the examples between trials.
    local learning_curve_score::Function = (function(size::Int64)
                                                dataset.examples = reduce(vcat, shuffle!(RandomDeviceInstance,
                                                                                        collect(dataset.examples[i, :]
                                                                                                for i in 1:size(dataset.examples)[1])));
                                                local training_set::AbstractMatrix = getindex(partition_dataset(dataset, 0, size), 1);
                                                return (1.0 - error_ratio(learner(dataset), dataset, training_set));
                                            end);


    local data_points::AbstractVector = collect((size, mean(learning_curve_score(size) for i in 1:trials))
                                                for size in sizes);
    dataset.examples = original_set;
    return data_points;
end

# The following variables are DataSets read from the aima-data repository.
orings_dataset = DataSet(name="orings", examples="./aima-data/orings.csv", target="Distressed",
                        attributes_names="Rings Distressed Temperature Pressure Flightnumber");

zoo_dataset = DataSet(name="zoo", examples="./aima-data/zoo.csv", target="type", exclude=["name"],
                    attributes_names=["name", "hair", "feathers", "eggs", "milk", "airborne",
                                    "aquatic", "predator", "toothed", "backbone", "breathes",
                                    "venomous", "fins", "legs", "tail", "domestic", "catsize", "type"]);

iris_dataset = DataSet(name="iris", examples="./aima-data/iris.csv");

"""
    RestaurantDataSet(;examples::Union{Void, String, AbstractMatrix}=nothing)

Return a new DataSet based on the restaurant waiting examples (Fig. 18.3).
"""
function RestaurantDataSet(;examples::Union{Void, String, AbstractMatrix}=nothing)
    if (typeof(examples) <: Void)
        return DataSet(name="restaurant", target="Wait", examples="./aima-data/restaurant.csv",
                        attributes_names=["Alternate", "Bar", "Fri/Sat", "Hungry", "Patrons", "Price",
                                        "Raining", "Reservation", "Type", "WaitEstimate", "Wait"]);
    else

        return DataSet(name="restaurant", target="Wait", examples=examples,
                        attributes_names=["Alternate", "Bar", "Fri/Sat", "Hungry", "Patrons", "Price",
                                        "Raining", "Reservation", "Type", "WaitEstimate", "Wait"]);
    end
end

restaurant_dataset = RestaurantDataSet();

# The decision tree for deciding whether to wait for a table (Fig. 18.2).
waiting_decision_tree = DecisionForkNode(attribute_index(restaurant_dataset, "Patrons"),
                                    attribute_name="Patrons",
                                    branches=Dict([Pair("None", DecisionLeafNode("No")),
                                                    Pair("Some", DecisionLeafNode("Yes")),
                                                    Pair("Full", DecisionForkNode(attribute_index(restaurant_dataset, "WaitEstimate"),
                                                                            attribute_name="WaitEstimate",
                                                                            branches=Dict([Pair(">60", DecisionLeafNode("No")),
                                                                                            Pair("0-10", DecisionLeafNode("Yes")),
                                                                                            Pair("30-60", DecisionForkNode(attribute_index(restaurant_dataset, "Alternate"),
                                                                                                                        attribute_name="Alternate",
                                                                                                                        branches=Dict([Pair("No", DecisionForkNode(attribute_index(restaurant_dataset, "Reservation"),
                                                                                                                                                                attribute_name="Reservation",
                                                                                                                                                                branches=Dict([Pair("Yes", DecisionLeafNode("Yes")),
                                                                                                                                                                            Pair("No", DecisionForkNode(attribute_index(restaurant_dataset, "Bar"),
                                                                                                                                                                                                    attribute_name="Bar",
                                                                                                                                                                                                    branches=Dict([Pair("No", DecisionLeafNode("No")),
                                                                                                                                                                                                                    Pair("Yes", DecisionLeafNode("Yes"))])))]))),
                                                                                                                                        Pair("Yes", DecisionForkNode(attribute_index(restaurant_dataset, "Fri/Sat"),
                                                                                                                                                                    attribute_name="Fri/Sat",
                                                                                                                                                                    branches=Dict([Pair("No", DecisionLeafNode("No")),
                                                                                                                                                                                    Pair("Yes", DecisionLeafNode("Yes"))])))]))),
                                                                                            Pair("10-30", DecisionForkNode(attribute_index(restaurant_dataset, "Hungry"),
                                                                                                                        attribute_name="Hungry",
                                                                                                                        branches=Dict([Pair("No", DecisionLeafNode("Yes")),
                                                                                                                                        Pair("Yes", DecisionForkNode(attribute_index(restaurant_dataset, "Alternate"),
                                                                                                                                                                attribute_name="Alternate",
                                                                                                                                                                branches=Dict([Pair("No", DecisionLeafNode("Yes")),
                                                                                                                                                                                Pair("Yes", DecisionForkNode(attribute_index(restaurant_dataset, "Raining"),
                                                                                                                                                                                                        attribute_name="Raining",
                                                                                                                                                                                                        branches=Dict([Pair("No", DecisionLeafNode("No")),
                                                                                                                                                                                                                        Pair("Yes", DecisionLeafNode("Yes"))])))])))])))])))]));

"""
    SyntheticRestaurantDataSet(n::Int64)
    SyntheticRestaurantDataSet()

Return a new restaurant dataset with 'n' examples (generates 20 examples when not
given a specific value).
"""
function SyntheticRestaurantDataSet(n::Int64)
    local examples_array::AbstractVector = [];
    for i in 1:n
        local new_example::AbstractVector = Array{Any, 1}(map((function(col::AbstractVector)
                                                                    return rand(RandomDeviceInstance, col);
                                                                end),
                                                                restaurant_dataset.values));
        new_example[restaurant_dataset.target] = classify(waiting_decision_tree, new_example);
        push!(examples_array, reshape(new_example, (1, length(new_example))));
    end
    return RestaurantDataSet(examples=Array{Any, 2}(matrix_vcat(examples_array...)));
end

function SyntheticRestaurantDataSet()
    local examples_array::AbstractVector = [];
    for i in 1:20
        local new_example::AbstractVector = Array{Any, 1}(map((function(col::AbstractVector)
                                                                    return rand(RandomDeviceInstance, col);
                                                                end),
                                                                restaurant_dataset.values));
        new_example[restaurant_dataset.target] = classify(waiting_decision_tree, new_example);
        push!(examples_array, reshape(new_example, (1, length(new_example))));
    end
    return RestaurantDataSet(examples=Array{Any, 2}(matrix_vcat(examples_array...)));
end

# The following artificial DataSets are generated randomly.
"""
    MajorityDataSet(k::Int64, n::Int64)

Return a DataSet of n k-bit examples for a majority problem. 'k' random bits are generated
randomly for each example. The target attribute is 1 if more than half the 'k' bits are 1,
otherwise the target attribute is 0.
"""
function MajorityDataSet(k::Int64, n::Int64)
    local examples_array::AbstractVector = [];
    for i in 1:n
        local bit_domain::AbstractVector = [0, 1];
        local bits::AbstractVector = collect(rand(RandomDeviceInstance, bit_domain) for i in 1:k);
        push!(bits, Int64(sum(bits)>(k/2)));
        push!(examples_array, reshape(bits, (1, length(bits))));
    end
    return DataSet(name="majority", examples=Array{Any, 2}(matrix_vcat(examples_array...)));
end

"""
    ParityDataSet(k::Int64, n::Int64; name::String="parity")

Return a DataSet of n k-bit examples for a parity problem. 'k' random bits are generated
randomly for each example. The target attribute is 1 if an odd amount of 'k' bits are 1,
otherwise the target attribute is 0.
"""
function ParityDataSet(k::Int64, n::Int64; name::String="parity")
    local examples_array::AbstractVector = [];
    for i in 1:n
        local bit_domain::AbstractVector = [0, 1];
        local bits::AbstractVector = collect(rand(RandomDeviceInstance, bit_domain) for i in 1:k);
        push!(bits, Int64(sum(bits) % 2));
        push!(examples_array, reshape(bits, (1, length(bits))));
    end
    return DataSet(name=name, examples=Array{Any, 2}(matrix_vcat(examples_array...)));
end

"""
    XorDataSet(n::Int64)

Return a DataSet of n 2-bit examples where 2 random bits are generated randomly for each
example. The target attribute is 1 only if 1 of the randomly generated bits is 1, otherwise
the target attribute is 0.
"""
function XorDataSet(n::Int64)
    return ParityDataSet(2, n, name="xor");
end

"""
    ContinuousXorDataSet(n::Int64)

Return a DataSet where each example consists of 2 random floats that are uniformly generated
in [0, 2). The target attribute is the xor of the floor of floats casted as integers.
"""
function ContinuousXorDataSet(n::Int64)
    local examples_array::AbstractVector = [];
    for i in 1:n
        local x::Float64;
        local y::Float64;
        x, y = (rand(RandomDeviceInstance) * 2, rand(RandomDeviceInstance) * 2);
        local example::AbstractVector = Array{Any, 1}();
        push!(example, x);
        push!(example, y);
        push!(example, (Int64(floor(x)) != Int64(floor(y))));
        push!(examples_array, reshape(example, (1, length(example))));
    end
    return DataSet(name="continuous xor", examples=Array{Any, 2}(matrix_vcat(examples_array...)));
end

