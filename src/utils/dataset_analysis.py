import numpy as np

class MultilabelDatasetCharacteristicsMetrics:
    def __init__(self, labels):
        """
        Initialize with labels, where `labels` is a list of lists,
        each list containing binary indicators (0 or 1) for the presence of each label.
        """
        self.labels = np.array(labels)  # Convert labels to a NumPy array for easier handling
        self.n = len(labels)  # Total number of instances

        # Proper check for an empty array
        if self.labels.size == 0:
            print("Warning: 'labels' is empty or not properly formatted. Please check the input data.")
        self.k = self.labels.shape[1] if self.labels.ndim > 1 and self.labels.size > 0 else 0  # Number of labels
        self.LSet = len(set(map(tuple, self.labels))) if self.labels.size > 0 else 0  # Number of distinct label sets

    # Additional methods remain unchanged

    def label_cardinality(self):
        """Calculate the average number of active labels per instance (Card)."""
        total_active_labels = self.labels.sum(axis=1).mean()
        return total_active_labels

    def label_density(self):
        """Calculate the normalized label cardinality (Dens)."""
        card = self.label_cardinality()
        dens = card / self.k if self.k else 0
        return dens

    def percentage_single_labeled_instances(self):
        """Calculate the percentage of instances with exactly one label (Pmin)."""
        single_label_count = (self.labels.sum(axis=1) == 1).sum()
        pmin = (single_label_count / self.n) * 100 if self.n else 0
        return pmin

    def label_diversity(self):
        """Calculate the diversity of label sets (Div)."""
        div = self.LSet / self.n if self.n else 0
        return div

    def calculate_scumble(self):
        """
        Calculate the SCUMBLE value for a multi-label dataset.
        
        Returns:
        float: The SCUMBLE value.
        """
        label_frequencies = self.labels.sum(axis=0)
        frequency_median = np.median(label_frequencies)
        minority_indices = np.where(label_frequencies < frequency_median)[0]
        majority_indices = np.where(label_frequencies >= frequency_median)[0]
        co_occurrence_matrix = np.dot(self.labels.T, self.labels)
        minority_majority_co_occurrence = co_occurrence_matrix[np.ix_(minority_indices, majority_indices)]
        minority_occurrences = label_frequencies[minority_indices][:, np.newaxis]
        normalized_concurrence = minority_majority_co_occurrence / minority_occurrences
        scumble_value = np.sum(normalized_concurrence) / (len(minority_indices) * len(majority_indices))
        return scumble_value


class MultilabelImbalanceAnalysis:
    """
    A class to analyze label imbalance in multilabel datasets, providing separate properties
    for each metric: IRLbl, MeanIR, and CVIR.
    """
    def __init__(self, labels):
        """
        Initialize with labels, where `labels` is a binary matrix (n_samples, n_labels)
        indicating the presence or absence of each label.
        
        Parameters:
        - labels (np.ndarray or list of lists): Binary label matrix for the dataset.
        """
        self.labels = np.array(labels) if isinstance(labels, list) else labels
        self.label_frequencies = self.labels.sum(axis=0)
        self.max_frequency = self.label_frequencies.max()

    @property
    def IRLbl(self):
        """
        Calculate and return the Imbalance Ratio per Label (IRLbl).
        
        Returns:
        - np.ndarray: IRLbl values for each label.
        """
        return self.max_frequency / self.label_frequencies

    @property
    def MeanIR(self):
        """
        Calculate and return the Mean Imbalance Ratio (MeanIR).
        
        Returns:
        - float: Mean of the IRLbl values.
        """
        return self.IRLbl.mean()

    @property
    def CVIR(self):
        """
        Calculate and return the Coefficient of Variation of IRLbl (CVIR).
        
        Returns:
        - float: CVIR value.
        """
        return self.IRLbl.std() / self.MeanIR


def calculate_positive_samples_per_label(labels):
    """
    Calculate the number of positive samples for each label in a multilabel dataset.

    Parameters:
    - labels (np.ndarray): A binary matrix (n_samples, n_labels) indicating the presence (1) or absence (0) of each label.

    Returns:
    - np.ndarray: An array containing the count of positive samples for each label.
    """
    # Ensure the input is a numpy array
    labels = np.array(labels)
    
    # Sum along columns to get the count of positive samples for each label
    positive_samples = labels.sum(axis=0)
    
    return positive_samples
