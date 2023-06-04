import torchvision.transforms as transforms
import torch.optim as optim
import random
import numpy as np
import timm

from model.models import *

if __name__ == "__main__":
    model_num = 10                      # total number of models (dinov2_vitl14: 1, resnet101: 10)
    total_epoch = 50                    # total epoch (dinov2_vitl14: 20, resnet101: 50)
    lr = 0.01                           # initial learning rate (dinov2_vitl14: 0.001, resnet101: 0.01)
    teacher_model_name = 'resnet101'    # dinov2_vitl14 or resnet101
    experiment = 'teacher'

    for s in range(model_num):
        # fix random seed
        seed_number = s
        random.seed(seed_number)
        np.random.seed(seed_number)
        torch.manual_seed(seed_number)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Check if GPU is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)

        # Define the data transforms
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8,
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Load the CIFAR-10 dataset
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=16)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=16)

        # Define the teacher model with pre-trained weights
        if teacher_model_name == 'dinov2_vitg14':
            model = teacher(model_name=teacher_model_name, num_classes=10, pretrained=True)
        elif teacher_model_name == 'resnet101':
            model = timm.create_model(teacher_model_name, pretrained=True, num_classes=10)
        model = model.to(device)  # Move the model to the GPU

        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
            model_without_dp = model.module
        else:
            model_without_dp = model

        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model_without_dp.parameters(), lr=lr, momentum=0.9)
        # Define the learning rate scheduler
        if teacher_model_name == 'dinov2_vitg14':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.1)
        elif teacher_model_name == 'resnet101':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

        def train():
            model.train()
            running_loss = 0.0

            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)  # Move the input data to the GPU
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i % 100 == 99:
                    print('[%d, %5d] lr: %.7f loss: %.3f' % (epoch + 1, i + 1, optimizer.param_groups[0]['lr'], running_loss / 100))
                    running_loss = 0.0

        def test():
            model.eval()

            # Test the model
            correct = 0
            total = 0
            with torch.no_grad():
                for data in testloader:
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)  # Move the input data to the GPU
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print('Accuracy of the network on the 10000 test images: %f %%' % (100 * correct / total))

        # Train the model
        for epoch in range(total_epoch):
            train()
            test()
            scheduler.step()

        print('Finished Training')

        # Save the checkpoint of the last model
        PATH = './weights/%s/%s_cifar10_%f_%d.pth' % (experiment, teacher_model_name, lr, seed_number)
        torch.save(model_without_dp.state_dict(), PATH)