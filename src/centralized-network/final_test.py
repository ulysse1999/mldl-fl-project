from resnet50 import ResNet
import torch
from data.provider import get_testing_data

path = "checkpoint.pt"

# if something went wrong with final_train, you can reload the model to evaluate it

def test(normalization):

    best_trained_model = ResNet(normalization)

    model_state, _optimizer_state = torch.load(path)
    best_trained_model.load_state_dict(model_state)
    print("Model loaded")

    best_trained_model.eval()
    
    testloader = get_testing_data()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to('cpu'), labels.to('cpu')
            outputs = best_trained_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    
    print(f"Test set accuracy : {correct/total}")

if __name__=="__main__":
    test(normalization="batch")
