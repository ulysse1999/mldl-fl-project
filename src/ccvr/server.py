from resnet50 import ResNet
import torch
from data.provider import get_testing_data
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import BatchSampler
from classifier import Classifier


CHECKPOINT_PATH = "global_checkpoint.pt"
path = CHECKPOINT_PATH

class Server:

    def __init__(self, normalization):        
        self.model = ResNet(normalization)

    def update_model(self, state_dict):
        self.model.load_state_dict(state_dict)

    def get_model_dict(self):
        return self.model.state_dict()

    def test_global(self):

        self.model.eval()

        testloader = get_testing_data(BATCH_SIZE=10000) # test dataset size is 10k

        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to('cpu'), labels.to('cpu')
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()


        print(f"Test set accuracy (global model) : {correct/total}")

    def save_model(self):

        torch.save(self.model.state_dict(), path)

        print("Model saved")

    def freeze_model(self):
        for name, param in self.model.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = False

    def unfreeze_model(self):
        for name, param in self.model.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = True

    def train_on_vr(self, means, covs, nc, n_training_iterations=200):
        """
        training on virtual representations 
        """
        total_samples = sum(nc.values())
        # for now we simply train for n_training_iterations each for each class

        self.freeze_model()

        for label in range(0,10):
            distrib = MultivariateNormal(loc=means[label], covariance_matrix=covs[label])
            samples = distrib.sample_n(n_training_iterations)

            self._train(samples, label)


        self.unfreeze_model()
            
    def _train(self, samples, label, batch_size=32):
        

        data = BatchSampler(samples, batch_size=batch_size, drop_last=True)

        criterion = CrossEntropyLoss()

        optimizer = SGD(self.model.parameters(), lr=1e-3, weight_decay=5e-4)

        model = Classifier()
        model.cuda()
        model.fc = self.model.fc


        for feats in data:
            
            optimizer.zero_grad()
            pred = model(torch.stack(feats))
            loss=criterion(pred, torch.tensor([label]*batch_size))
            loss.backward()
            optimizer.step()

            #f=f.cuda()
            
        model.cpu()
        self.model.fc = model.fc
