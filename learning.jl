
import Base: getindex;

export euclidean_distance, mean_square_error, root_mean_square_error,
        mean_error, manhattan_distance, mean_boolean_error, hamming_distance, gaussian,
        DataSet, set_problem, attribute_index, check_dataset_fields,
        check_example, update_values, add_example, remove_examples, sanitize, summarize,
        classes_to_numbers, split_values_by_classes, find_means_and_deviations,
        CountingProbabilityDistribution, add, smooth_for_observation, getindex, top, sample,
        PluralityLearner, predict,
        AbstractNaiveBayesModel, NaiveBayesLearner,
        NaiveBayesDiscreteModel, NaiveBayesContinuousModel, NearestNeighborLearner,
        DecisionLeafNode, DecisionForkNode, classify;

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
                    examples::Union{Void, String, AbstractVector}=nothing,
                    values::Union{Void, AbstractVector}=nothing,
                    inputs::Union{Void, String, AbstractVector}=nothing, target::Int64=-1,
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

function attribute_index(ds::DataSet, attribute::String)
    return utils.index(ds.attributes, attribute);
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

Return a copy of the given array 'example', such that non-input attributes are replaced with 'nothing'.
"""
function sanitize(ds::DataSet, example::AbstractVector)
    local sanitized_example::AbstractVector = [];
    for (i, example_item) in enumerate(example)
        if (i in ds.inputs)
            push!(sanitized_example, example_item)
        else
            push!(sanitized_example, nothing);
        end
    end
    return sanitized_example;
end

function sanitize(ds::DataSet, example::AbstractMatrix)
    local sanitized_example::AbstractMatrix = copy(example);
    for i in size(example)[1]
        for j in size(example)[2]
            if (!(example[i, j] in ds.inputs))
                example[i, j] = nothing;
            end
        end
    end
    return sanitized_example;
end

function classes_to_numbers(ds::DataSet, classes::Void)
    local new_classes::AbstractVector = sort(ds.values[ds.target]);
    for example in (ds.examples[i, :] for i in 1:size(ds.examples)[1])
        local index_val::Int64 = utils.index(new_classes, example[ds.target]);
        if (index_val == -1)
            error("classes_to_numbers(): Could not find ", example[ds.target], " in ", new_classes, "!");
        end
        example[ds.target] = index_val;
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
    for example in (ds.examples[i, :] for i in 1:size(ds.examples)[1])
        local index_val::Int64 = utils.index(new_classes, example[ds.target]);
        if (index_val == -1)
            error("classes_to_numbers(): Could not find ", example[ds.target], " in ", new_classes, "!");
        end
        example[ds.target] = index_val;
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

#=

    CountingProbabilityDistribution is a probability distribution for counting

    observations. Unlike the other implementations of AbstractProbabilityDistribution,

    CountingProbabilityDistribution calculates the key's probability when

    accessing the CountingProbabilityDistribution by the given key.

=#
type CountingProbabilityDistribution
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

function add(cpd::CountingProbabilityDistribution, observation)
    smooth_for_observation(cpd, observation);
    cpd.dict[observation] = cpd.dict[observation] + 1;
    cpd.number_of_observations = cpd.number_of_observations + 1;
    cpd.sample_function = Nullable{Function}();
    nothing;
end

function smooth_for_observation(cpd::CountingProbabilityDistribution, observation)
    if (!(observation in keys(cpd.dict)))
        cpd.dict[observation] = cpd.default;
        cpd.number_of_observations = cpd.number_of_observations + cpd.default;
        cpd.sample_function = Nullable{Function}();
    end
    nothing;
end

"""
    getindex(cpd::CountingProbabilityDistribution, key)

Return the probability of the given 'key'.
"""
function getindex(cpd::CountingProbabilityDistribution, key)
    smooth_for_observation(cpd, key);
    return (Float64(cpd.dict[key]) / Float64(cpd.number_of_observations));
end

"""
    top(cpd::CountingProbabilityDistribution, n::Int64)

Return an array of (observation_count, observation) tuples such that the array
does not exceed length 'n'.
"""
function top(cpd::CountingProbabilityDistribution, n::Int64)
    local observations::AbstractVector = sort(collect(reverse((i...)) for i in cpd.dict),
                                                lt=(function(p1::Tuple{Number, Any}, p2::Tuple{Number, Any})
                                                        return (p1[1] < p2[1]);
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
    return cpd.sample_function();
end

type PluralityLearner{T}
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

type NaiveBayesLearner
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
type NearestNeighborLearner
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

type DecisionLeafNode{T}
    result::T

    function DecisionLeafNode{T}(result::T)
        return new(result);
    end
end

function classify(dl::DecisionLeafNode, example::AbstractVector)
    return dl.result;
end

DecisionLeafNode(result) = DecisionLeafNode{typeof(result)}(result);

type DecisionForkNode
    attribute::Int64
    attribute_name::Nullable
    default_child::Nullable{DecisionLeafNode}
    branches::Dict

    function DecisionForkNode(attribute::Int64;
                        attribute_name::Union{String, Void}=nothing,
                        default_child::Union{DecisionLeafNode, Void}=nothing,
                        branches::Union{Dict, Void}=nothing)
        local new_attribute_name::Nullable;
        local new_branches::Dict;
        if (typeof(attribute_name) <: Void)
            new_attribute_name = Nullable{Int64}(attribute);
        else
            new_attribute_name = Nullable{String}(attribute_name);
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
    local attribute_value::Float64 = example[df.attribute];
    if (haskey(df.branches, attribute_value))
        return classify(df.branches[attribute_value], example);
    else
        return classify(df.default_child, example);
    end
end

