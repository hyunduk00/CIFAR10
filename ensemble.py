import torch
import torchvision
import torchvision.transforms as transforms
import timm
from model.models import *


if __name__ == "__main__":
    model_num = 10                      # student: 10, teacher: 1, proposed: 10
    lr = 0.01                           # student: 0.01, teacher: dinov2_vitl14=0.001, resnet101=0.01, proposed: 0.01
    experiment = 'proposed'             # student, teacher, proposed
    model_name = 'resnet101_resnet18'   # student: resnet18, teacher: dinov2_vitl14 or resnet101, proposed: dinov2_vitl14_resnet18 or resnet101_resnet18

    for s in range(model_num):
        # Check if GPU is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)

        # Define the data transforms
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.TenCrop(224),
            transforms.Lambda(lambda crops: torch.stack([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(transforms.ToTensor()(crop)) for crop in crops]))
        ])

        # Load the CIFAR-10 test dataset
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=16)

        # Define the list of models for ensemble
        models = []
        for i in range(s+1):
            # Define the model with pre-trained weights
            if model_name == 'resnet18' or model_name == 'resnet101':
                model = timm.create_model(model_name, num_classes=10)
            elif model_name == 'dinov2_vitl14':
                model = teacher(model_name=model_name, num_classes=10, pretrained=False)
            elif model_name == 'dinov2_vitl14_resnet18' or model_name == 'resnet101_resnet18':
                model = timm.create_model('resnet18', num_classes=10)

            model.load_state_dict(torch.load(f"./weights/%s/%s_cifar10_%f_%d.pth" % (experiment, model_name, lr, i)))  # Load the trained weights
            model.eval()  # Set the model to evaluation mode
            model = model.to(device)  # Move the model to the GPU
            models.append(model)

        # Evaluate the ensemble of models
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)  # Move the input data to the GPU
                bs, ncrops, c, h, w = images.size()
                outputs = torch.zeros(bs, 10).to(device)  # Initialize the output tensor with zeros
                for model in models:
                    model_output = model(images.view(-1, c, h, w))  # Reshape the input to (bs*10, c, h, w)
                    model_output = model_output.view(bs, ncrops, -1).mean(1)  # Average the predictions of the 10 crops
                    outputs += model_output
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the %d ensemble on the 10000 test images: %f %%' % (s+1, 100 * correct / total))