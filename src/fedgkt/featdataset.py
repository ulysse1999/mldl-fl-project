from torch.utils.data import Dataset

class FeatureDataset(Dataset):

    def __init__(self, features, predictions):

        self.feats = features
        self.predictions = predictions

    def __len__(self):

        return len(self.feats)

    def __getitem__(self, idx):

        return self.feats[idx], self.predictions[idx]

