from resnet50 import ResNet
import torch
from data.provider import get_testing_data


CHECKPOINT_PATH = "global_checkpoint.pt"
path = CHECKPOINT_PATH

class Server:

    def __init__(self, normalization):        
        self.model = ResNet(normalization)

    def update_model(self, state_dict):
        
        self.model.load_state_dict(state_dict)

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
