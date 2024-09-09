import math
# Function to extract metadata statistics
def extract_metadata_statistics(metadata):
    # Number of observations (same for any label as 'total' is the same)
    number_of_observations = metadata['class_counts']['a']['total']

    # Number of classes
    number_of_classes = len(metadata['class_counts'])

    # Total number of positive instances across all labels
    total_positives = sum(label_info['positive'] for label_info in metadata['class_counts'].values())

    # Total number of negative instances across all labels
    total_negatives = sum(label_info['negative'] for label_info in metadata['class_counts'].values())

    return number_of_observations, number_of_classes, total_positives, total_negatives

# Function to compute class frequencies and imbalance
def compute_class_analysis(metadata, number_of_observations):
    label_frequencies = {}
    detailed_label_analysis = {}

    for label, label_info in metadata['class_counts'].items():
        positive_instances = label_info['positive']
        negative_instances = label_info['negative']
        total_instances = positive_instances + negative_instances

        # Calculate label frequency
        label_frequencies[label] = positive_instances / number_of_observations

        # Store detailed metrics for each label
        detailed_label_analysis[label] = {
            'total_positive': positive_instances,
            'total_negative': negative_instances,
            'percentage_positive': (positive_instances / total_instances) \
                * 100 if total_instances != 0 else 0,
            'percentage_negative': (negative_instances / total_instances) \
                * 100 if total_instances != 0 else 0
        }

    return label_frequencies, detailed_label_analysis

# Function to display results
def display_analysis_results(number_of_observations, number_of_classes, \
                             total_positives, total_negatives, label_frequencies, detailed_label_analysis):
    # Average number of labels per instance
    average_labels_per_instance = total_positives / number_of_observations

    # Average number of instances per label
    average_instances_per_label = total_positives / number_of_classes

    # Most frequent and least frequent labels
    most_frequent_label = max(label_frequencies, key=label_frequencies.get)
    least_frequent_label = min(label_frequencies, key=label_frequencies.get)

    # Percentage of positive instances in the dataset
    percentage_positive_instances = (total_positives / (total_positives + total_negatives)) * 100

    # Print summary of the dataset
    print("Number of Observations:", number_of_observations)
    print("Number of Classes:", number_of_classes)
    print("Average Number of Labels per Instance:", average_labels_per_instance)
    print("Average Number of Instances per Label:", average_instances_per_label)
    print("Most Frequent Label:", most_frequent_label)
    print("Least Frequent Label:", least_frequent_label)
    print("Percentage of Positive Instances:", percentage_positive_instances)

    # Print detailed label analysis
    for label, metrics in detailed_label_analysis.items():
        print(f"\nLabel: {label}")
        print(f"  Total Positive Instances: {metrics['total_positive']}")
        print(f"  Total Negative Instances: {metrics['total_negative']}")
        print(f"  Percentage Positive Instances: {metrics['percentage_positive']:.2f}%")
        print(f"  Percentage Negative Instances: {metrics['percentage_negative']:.2f}%")

# Main pipeline function
def dataset_class_imbalance(metadata):
    number_of_observations, number_of_classes, total_positives, total_negatives = extract_metadata_statistics(metadata)
    label_frequencies, detailed_label_analysis = compute_class_analysis(metadata, number_of_observations)
    display_analysis_results(number_of_observations, number_of_classes, total_positives, \
                             total_negatives, label_frequencies, detailed_label_analysis)

# # Execute the pipeline with the provided metadata
# main_pipeline(metadata)