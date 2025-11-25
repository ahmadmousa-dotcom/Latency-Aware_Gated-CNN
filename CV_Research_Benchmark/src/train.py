import torch
import torchvision
import torchvision.transforms as transforms
from .config import Config
from .utils import get_temperature

def get_dataloaders(dataset_name):
    print(f"[Data] Preparing {dataset_name}...")
    
    if dataset_name == 'SVHN':
        # SVHN has different normalization stats and split names
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.437, 0.443, 0.472), (0.198, 0.201, 0.197))
        ])
        trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform)
        testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform)
        
    elif dataset_name == 'CIFAR10':
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.491, 0.482, 0.446), (0.202, 0.199, 0.201))
        ])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        
    else: # CIFAR100
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.507, 0.486, 0.440), (0.267, 0.256, 0.276))
        ])
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS)
    testloader = torch.utils.data.DataLoader(testset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)
    
    return trainloader, testloader

def train_engine(model, dataset_name, name="Model"):
    model = model.to(Config.DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=Config.LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
    criterion = torch.nn.CrossEntropyLoss()
    
    train_loader, _ = get_dataloaders(dataset_name)
    history = {'acc': [], 'sparsity': []}
    
    print(f"\n>>> Training {name} on {dataset_name}")
    
    for epoch in range(Config.EPOCHS):
        model.train()
        temp = get_temperature(epoch)
        
        correct = 0
        total = 0
        active_sum = 0
        gate_count = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
            optimizer.zero_grad()
            
            outputs, gates = model(inputs, temperature=temp)
            loss = criterion(outputs, labels)
            
            if len(gates) > 0:
                reg = sum(torch.sum(g) for g in gates)
                loss += Config.LAMBDA_LATENCY * reg
                
                # Stats
                for g in gates:
                    active_sum += torch.sum(g).item()
                    gate_count += g.numel()
            
            loss.backward()
            optimizer.step()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        scheduler.step()
        
        acc = 100 * correct / total
        sparsity = (active_sum / gate_count) if gate_count > 0 else 1.0
        history['acc'].append(acc)
        history['sparsity'].append(sparsity)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{Config.EPOCHS} | Acc: {acc:.2f}% | Density: {sparsity:.2f}")
            
    return history, model