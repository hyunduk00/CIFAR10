import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import timm
import random
import numpy as np

from model.models import *


class DistillationLoss(nn.Module):
    def __init__(self, tau, alpha, total_epoch):
        super().__init__()
        self.tau = tau
        self.alphas = np.linspace(alpha, 0, total_epoch)
        self.alpha = alpha
        self.last_epoch = -1

    def forward(self, student_outputs, labels, teacher_outputs):
        student_loss = F.cross_entropy(input=student_outputs, target=labels)
        distillation_loss = F.kl_div(F.log_softmax(student_outputs/self.tau, dim=1),
                             F.softmax(teacher_outputs/self.tau, dim=1), reduction='batchmean') * (self.tau * self.tau)
        loss = (1-self.alpha) * student_loss + self.alpha * distillation_loss

        return loss, student_loss, distillation_loss

    def step(self, epoch=None):
        if epoch is None:
            self.last_epoch +=1
            self.alpha = self.alphas[self.last_epoch]
        else:
            self.last_epoch  = epoch
            self.alpha = self.alphas[self.last_epoch]


if __name__ == "__main__":
    # initialize parameters
    model_num = 10          # total number of models
    total_epoch = 50        # total epoch
    lr = 0.01               # initial learning rate
    student_lr = 0.01       # pretrained learning rate for student network
    teacher_lr = 0.001       # pretrained learning rate for teacher network (dinov2_vitl14: 0.001, resnet101: 0.01)
    alpha = 0.9             # for distillation loss
    tau = 10
    teacher_model_name = 'dinov2_vitl14'    # dinov2_vitl14 or resnet101
    student_model_name = 'resnet18'
    experiment = 'proposed'
    teacher_experiment = 'teacher'
    student_experiment = 'student'

    for s in range(7, model_num):
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

        # # Define the student model with pre-trained weights
        if teacher_model_name == 'resnet101':
            teacher_model = timm.create_model(teacher_model_name, num_classes=10)
            teacher_model.load_state_dict(torch.load('./weights/%s/%s_cifar10_%f_%d.pth' % (teacher_experiment, teacher_model_name, teacher_lr, s)))  # Load the trained weights
        elif teacher_model_name == 'dinov2_vitl14':
            teacher_model = teacher(model_name=teacher_model_name, num_classes=10, pretrained=False)
            teacher_model.load_state_dict(torch.load('./weights/%s/%s_cifar10_%f_0.pth' % (teacher_experiment, teacher_model_name, teacher_lr)))  # Load the trained weights

        teacher_model = teacher_model.to(device)  # Move the model to the GPU

        student_model = timm.create_model('resnet18', num_classes=10)
        student_model.load_state_dict(torch.load(f"./weights/%s/resnet18_cifar10_%f_%d.pth" % (student_experiment, student_lr, s)))  # Load the trained weights
        student_model = student_model.to(device)  # Move the model to the GPU

        if torch.cuda.device_count() > 1:
            teacher_model = torch.nn.DataParallel(teacher_model)
            student_model = torch.nn.DataParallel(student_model)
            student_model_without_dp = student_model.module
        else:
            student_model_without_dp = student_model

        # Define the loss function and optimizer
        criterion = DistillationLoss(tau=tau, alpha=alpha, total_epoch=total_epoch)
        optimizer = optim.SGD(student_model_without_dp.parameters(), lr=lr, momentum=0.9)

        # Define the learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epoch, eta_min=1e-7)

        def train():
            teacher_model.eval()
            student_model.train()
            running_loss = 0.0
            running_stuendt_loss = 0.0
            running_distillation_loss = 0.0

            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)  # Move the input data to the GPU
                optimizer.zero_grad()
                student_outputs = student_model(inputs)
                with torch.no_grad():
                    teacher_outputs = teacher_model(inputs)

                loss, student_loss, distillation_loss = criterion(student_outputs, labels, teacher_outputs)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                running_stuendt_loss += student_loss.item()
                running_distillation_loss += distillation_loss.item()
                if i % 100 == 99:
                    print('[%d, %5d] lr: %.7f alpha: %.3f, loss: %.3f(student: %.3f, distillation: %.3f)' % (epoch + 1, i + 1, optimizer.param_groups[0]['lr'], criterion.alpha, running_loss / 100, running_stuendt_loss / 100, running_distillation_loss / 100))
                    running_loss = 0.0
                    running_stuendt_loss = 0.0
                    running_distillation_loss = 0.0

        def test():
            student_model.eval()

            # Test the model
            correct = 0
            total = 0
            with torch.no_grad():
                for data in testloader:
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)  # Move the input data to the GPU
                    outputs = student_model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print('Accuracy of the network on the 10000 test images: %f %%' % (100 * correct / total))

        # Train the model
        for epoch in range(total_epoch):
            criterion.step()
            train()
            test()
            scheduler.step()

        print('Finished %s-th Training' % (seed_number))

        # Save the checkpoint of the last model
        PATH = './weights/%s/%s_%s_cifar10_%f_%d.pth' % (experiment, teacher_model_name, student_model_name, lr, seed_number)
        torch.save(student_model_without_dp.state_dict(), PATH)

