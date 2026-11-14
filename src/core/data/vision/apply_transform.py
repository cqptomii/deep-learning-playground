from torch.utils.data import Dataset

class ApplyTransform(Dataset):
    """
    A wrapper class for applying transformations to a dataset.

    Args:
        dataset (Dataset): The original dataset to be wrapped.
        transform (callable, optional): A function/transform to apply to each sample in the dataset.
    """

    def __init__(self, dataset: Dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label