import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler

from icrt.util.misc import DistributedWeightedSubEpochSampler
from icrt.data.dataset import CustomConcatDataset

# Define two fake datasets
class FakeDataset(Dataset):
    def __init__(self, size, label):
        self.size = size
        self.label = label  # Identifies the dataset
        self.data = list(range(size))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {"data": self.data[idx], "label": self.label}

    def save_split(self, *args, **kwargs):
        # Stub method for compatibility
        pass

    def shuffle_dataset(self, seed=0):
        torch.manual_seed(seed)
        torch.random.shuffle(self.data)

# Test function
def test_concat_dataset_with_balanced_weights():
    # Create two fake datasets of different sizes
    dataset1 = FakeDataset(size=10, label=0)
    dataset2 = FakeDataset(size=300, label=1)

    # Combine the datasets
    concat_dataset = CustomConcatDataset([dataset1, dataset2])

    # Retrieve the balanced sampling weights
    weights = concat_dataset.balanced_sampling_weights

    # Create a sampler with the weights
    sampler = WeightedRandomSampler(weights, num_samples=20, replacement=True)

    # Create a DataLoader with the sampler
    dataloader = DataLoader(concat_dataset, sampler=sampler, batch_size=5)

    # Collect sampled labels to verify balance
    sampled_labels = []
    for batch in dataloader:
        sampled_labels.extend(batch["label"])

    # Count the occurrences of each label
    label_counts = {0: sampled_labels.count(0), 1: sampled_labels.count(1)}

    # Print results
    print(f"Balanced Sampling Weights: {weights}")
    print(f"Sampled Labels: {sampled_labels}")
    print(f"Label Counts: {label_counts}")

    # Verify sampling balance: Expect approximately equal sampling despite size difference
    assert label_counts[0] > 5, "Dataset1 should be sampled sufficiently."
    assert label_counts[1] > 5, "Dataset2 should be sampled sufficiently."

def test_distributed_weighted_sampler():
    # Create two datasets of different sizes
    dataset1 = FakeDataset(size=10, label=0)
    dataset2 = FakeDataset(size=99, label=1)

    # Combine the datasets
    concat_dataset = CustomConcatDataset([dataset1, dataset2])

    # Generate balanced weights
    weights = concat_dataset.balanced_sampling_weights
    # print(f"Balanced Sampling Weights:\n{weights}")
    # exit()

    # Number of replicas (e.g., distributed workers)
    num_replicas = 4

    # Run test for each rank
    rank_results = {}
    for rank in range(num_replicas):
        # Create the sampler for the current rank
        sampler = DistributedWeightedSubEpochSampler(
            dataset=concat_dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=True,
            split_epoch=1,
            seed=45,
            weights=weights,
        )

        # Create a DataLoader with the sampler
        dataloader = DataLoader(concat_dataset, sampler=sampler, batch_size=5)

        # Collect sampled indices and labels for this rank
        sampled_indices = []
        sampled_labels = []
        for batch in dataloader:
            sampled_indices.extend(batch["data"])
            sampled_labels.extend(batch["label"])

        rank_results[rank] = {
            "indices": sampled_indices,
            "labels": sampled_labels,
        }

    # Verify no overlap between ranks
    all_indices = []
    for rank in range(num_replicas):
        all_indices.extend(rank_results[rank]["indices"])
    assert len(all_indices) == len(set(all_indices)), "Overlap detected between ranks!"

    # Verify balanced sampling
    total_labels = []
    for rank in range(num_replicas):
        total_labels.extend(rank_results[rank]["labels"])
    label_counts = {label: total_labels.count(label) for label in set(total_labels)}

    print(f"Label Counts Across Ranks: {label_counts}")

    # # Expect approximately equal counts
    # assert abs(label_counts[0] - label_counts[1]) <= 2, "Labels are not balanced!"

    # Print results for debugging
    for rank, result in rank_results.items():
        print(f"Rank {rank}:")
        # print number of indices with label 0 and 1
        print(f"  Label 0: {result['labels'].count(0)}")
        print(f"  Label 1: {result['labels'].count(1)}")
        # print(f"  Indices: {result['indices']}")
        # print(f"  Labels: {result['labels']}")


# Run the test
test_concat_dataset_with_balanced_weights()
test_distributed_weighted_sampler()
