
export euclidean_distance, mean_square_error, root_mean_square_error,
        mean_error, manhattan_distance, mean_boolean_error, hamming_distance,
        DataSet, set_problem, attribute_index, check_dataset_fields,
        check_example, update_values, add_example, remove_examples, sanitize, summarize;

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

type DataSet
    name::String
    source::String
    examples::AbstractMatrix
    attributes::AbstractVector
    attributes_names::AbstractVector
    values::Nullable{AbstractVector}
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
        if (typeof(values) <: Void)
            values_is_set = false;
        else
            values_is_set = true;
        end

        # Construct new DataSet without 'inputs' and 'target' fields.
        local ds::DataSet = new(name, source, examples_array, attributes_array, attributes_names_array,
                                Nullable{AbstractVector}(values), exclude, distance);

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
    if (isnull(ds.values))
        update_values(ds);
    elseif (length(get(ds.values)) == 0)
        update_values(ds);
    end
    nothing;
end

function set_problem(ds::DataSet, target::Int64, inputs::AbstractVector, exclude::AbstractVector)
    ds.target = attribute_index(ds, target);
    ds.inputs = removeall(inputs, ds.target);
    if (isnull(ds.values))
        update_values(ds);
    elseif (length(get(ds.values)) == 0)
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
    ds.values = collect(Set(ds.examples[i, :] for i in 1:size(ds.examples)[1]));
    nothing;
end

function add_example(ds::DataSet, example::AbstractVector)
    check_example(ds, example);
    vcat(ds.examples, example);
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

function summarize(ds::DataSet)
    return @sprintf("<DataSet(%s): %d examples, %d attributes>", ds.name, length(ds.examples), length(ds.attributes));
end

