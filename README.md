# Reference
 - https://github.com/heechul-knu/cifar-baseline
 - https://github.com/facebookresearch/dinov2

# Requirement
 - pytorch: conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
 - timm: conda install -c conda-forge timm
 
# Run
- Baseline model
  ```
  python ./baseline/train_base
  python ./baseline/ensemble_base.py
  ```
- proposed model

  ```
  python train_student              # train student network
  python train_teacher.py           # train teacher network
  python knowledge distillation.py  # train proposed network using knowledge distillation
  python ensemble.py                # before test set to parameters for each task(student, teacher, proposed)
  ```


# Result
- Baseline model
  - Network: Resnet-18 with pretrained weights from ImageNet-1k
  - Augmentation
    - Train: Resize, RandomCrop, RandomHorizontalFlip, Normalize
    - Test: Resize TenCrop, Normalize
  - Parameters
    - Epochs = 50
    - Learning rate = 0.01
    - Number of models = 2
  - Train
    - Criterion = CrossEntropyLoss
    - Optimizer = SGD with momentum=0.9
    - Scheduler = StepLR with step_size=30, gamma=0.1
  - Accuracy
 
    | Ensemble |   1   |     2     |
    |:--------:|:-----:|:---------:|
    | accuracy | 96.74 | **97.04** |
  
- Student model
  - Network: ResNet-18 with pretrained weights from ImageNet-1k
  - Augmentation
    - Train: Resize, RandomCrop, RandomHorizontalFlip, **ColorJitter**, Normalize
    - Test: Resize TenCrop, Normalize
  - Parameters
    - Epochs = 50
    - Learning rate = 0.01
    - Number of models = **10**
  - Train
    - Criterion = CrossEntropyLoss
    - Optimizer = SGD with momentum=0.9
    - Scheduler = StepLR with step_size=30, gamma=0.1
  - Accuracy
  
    | model_num |   1   |  2    |  3    |  4    |  5    |  6    |  7    |  8    |  9    |    10     |
    |:---------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:---------:|
    | accuracy  | 96.84 | 97.09 | 97.28 | 97.38 | 97.34 | 97.41 | 97.42 | 97.45 | 97.50 | **97.50** |

- Teacher model-case1
  - Network
    - Backbone: **dinov2_vitl14** with pretrained weights from ImageNet-1k
    - Classifier: linear layer
  - Augmentation
    - Train: Resize, RandomCrop, RandomHorizontalFlip, ColorJitter, Normalize
    - Test: Resize TenCrop, Normalize
  - Parameters
    - Epochs = 20
    - Learning rate = 0.001
    - Number of models = 1
    - Freeze the parameters of backbone and only the parameters of classifier are trained
  - Train
    - Criterion = CrossEntropyLoss
    - Optimizer = SGD with momentum=0.9
    - Scheduler = StepLR with step_size=12, gamma=0.1
  - Accuracy: 99.24%
  

- Teacher model-case2
  - Network: **ResNet-101** with pretrained weights from ImageNet-1k
  - Augmentation
    - Train: Resize, RandomCrop, RandomHorizontalFlip, ColorJitter, Normalize
    - Test: Resize TenCrop, Normalize
  - Parameters
    - Epochs = 50
    - Learning rate = 0.01
    - Number of models = 10
  - Train
    - Criterion = CrossEntropyLoss
    - Optimizer = SGD with momentum=0.9
    - Scheduler = StepLR with step_size=30, gamma=0.1
  - Accuracy
  
    | model_num |   1   |   2   |   3   |   4   |   5   |   6   |   7   |     8     |   9   |  10   |
    |:---------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:---------:|:-----:|:-----:|
    | accuracy  | 97.82 | 98.20 | 98.24 | 98.19 | 98.19 | 98.22 | 98.24 | **98.33** | 98.30 | 98.25 |

- Proposed model  
  - Teacher network: pretrained from **CIFAR-10**
    - Backbone: dinov2_vitl14
    - Classifier: linear layer
  - Student network: ResNet-18 with pretrained weights from **CIFAR-10**
  - Parameters
    - Epochs = 50
    - Learning rate = 0.01
    - model_num = 10
  - Train
    - Criterion = **knowledge distillation**
      - alpha: 0.9 to 0 -> numpy.linspace(0.9, 0, Epochs)
      - tau: 10
    - Optimizer = SGD with momentum=0.9
    - Scheduler = **CosineAnnealingLR** with T_max=50, eta_min=1e-7
  - Accuracy
   
    | model_num |   1   |   2   |   3   |   4   |   5   |   6   |   7   |   8   |     9     |  10   |
    |:---------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:---------:|:-----:|
    | accuracy  | 97.05 | 97.15 | 97.33 | 97.36 | 97.16 | 97.24 | 97.24 | 97.29 | **97.37** | 97.36 |
  
  - Teacher network: ResNet-101 with pretrained weights from **CIFAR-10**
  - Student network: ResNet-18 with pretrained weights from **CIFAR-10**
  - Parameters
    - Epochs = 50
    - Learning rate = 0.01
    - model_num = 10
    - **knowledge distillation**
      - alpha: 0.9 to 0 (numpy.linspace(0.9, 0, 50))
      - tau: 10
  - Train
    - Criterion = **knowledge distillation**
      - alpha: 0.9 to 0 -> numpy.linspace(0.9, 0, Epochs)
      - tau: 10
    - Optimizer = SGD with momentum=0.9
    - Scheduler = **CosineAnnealingLR** with T_max=50, eta_min=1e-7
  - Accuracy
   
    | model_num |   1   |   2   |   3   |   4   |   5   |   6   |   7   |   8   |     9     |  10   |
    |:---------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:---------:|:-----:|
    | accuracy  | 97.28 | 97.31 | 97.38 | 97.42 | 97.41 | 97.38 | 97.40 | 97.41 | **97.43** | 97.39 |
  