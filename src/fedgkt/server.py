from torch.optim import SGD
from torch.nn import CrossEntropyLoss, KLDivLoss
from resnet49 import ResNet49
from data.provider import get_testing_data
import torch


CHECKPOINT_PATH = "global_checkpoint.pt"
path = CHECKPOINT_PATH

class Server:

    def __init__(self, normalization, epochs):        
        self.model = ResNet49(normalization)
        self.epochs = epochs

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


    def train(self, client_learnings):

        optimizer = SGD(self.model.parameters(), lr=1e-3, weight_decay=5e-4)

        crossEntropy = CrossEntropyLoss(reduction='sum')
        KLDiv = KLDivLoss(reduction='sum')
        crossEntropy.cuda()
        KLDiv.cuda()

        self.model.cuda()
        self.model.train()

        pred_list = []


        # training loop
        for epoch in range(self.epochs):
            for i, data in enumerate(client_learnings):

                imgs, cl_logit = data
                imgs, cl_logit = imgs.cuda(), cl_logit.cuda()
                
                print(imgs.size())

                optimizer.zero_grad()
                pred = self.model(imgs)
                if epoch==self.epochs-1:
                    pred_list.append(pred)
                pred = pred.cuda()
                
                loss = crossEntropy(pred, cl_logit.softmax(dim=1)).item() + KLDiv(pred.log(), cl_logit.softmax(dim=1)).item()
                loss.backward(retain_graph=True)
                optimizer.step()
                
        pred_list = torch.stack(pred_list)

        self.model = self.model.to('cpu')

        self.model_dict = self.model.state_dict()
        torch.cuda.empty_cache()

        return pred_list
