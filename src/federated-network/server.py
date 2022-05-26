from resnet50 import ResNet

CHECKPOINT_PATH = "global_checkpoint.pt"
path = CHECKPOINT_PATH

class Server:

    def __init__(self, normalization):        
        self.model = ResNet()

    def update_model(self, state_dict):
        
        model.load_state_dict(state_dict)

    def test_global(self):

        model.eval()

        testloader = get_testing_data()

        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to('cpu'), labels.to('cpu')
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()


        print(f"Test set accuracy (global model) : {correct/total}")

    def save_model(self):

        torch.save((model.state_dict(), optimizer.state_dict()), path)

        print("Model saved")
