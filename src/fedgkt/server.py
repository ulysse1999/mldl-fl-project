from torch.optim import SGD
from torch.nn import CrossEntropyLoss, KLDivLoss
from resnet49 import ResNet49
from data.provider import get_testing_data
import torch
from torch.autograd import detect_anomaly


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

        dataset = torch.utils.data.DataLoader(
            client_learnings,
            batch_size=10,
            shuffle=True,
            num_workers=0,
            pin_memory=False
            )


        # training loop
        for epoch in range(self.epochs):
            for i, data in enumerate(dataset):

                imgs, cl_logit, labels = data
                imgs, cl_logit, labels = imgs.cuda(), cl_logit.cuda(), labels.cuda()

                optimizer.zero_grad()



                with detect_anomaly():

                    cl_logit = cl_logit.softmax(dim=1)
                    
                    pred = self.model(imgs)
                    
                    normalized_pred = pred.softmax(dim=1)
                    
                    if epoch==self.epochs-1:
                        aux_pred = pred.detach()
                        pred_list.extend(aux_pred)

                    pred = pred.cuda()

                    
                    loss = crossEntropy(pred, labels) + KLDiv(normalized_pred, cl_logit)

                    loss.backward()
                    
                    optimizer.step()
                

        self.model = self.model.to('cpu')

        self.model_dict = self.model.state_dict()
        torch.cuda.empty_cache()

        return pred_list
