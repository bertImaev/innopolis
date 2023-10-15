import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os
import time
from PIL import Image


def classify_image(image_path):
    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    image = preprocess(image).unsqueeze(0)
    image = image.to(device)

    model.eval()
    with torch.no_grad():
        out = model(image)
        _, predicted = torch.max(out, 1)

    return class_names[predicted[0]]


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]),
}

data_dir = r'D:\01-Учеба Иннополис 2023\Домашки\LabCV(47)\DataSet'
batch_size = 32
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.CrossEntropyLoss()
lr_var = [0.001, 0.01, 0.05]
momentum_var = [0.9, 1, 0.5]
epochs_var = [5, 7, 10]
path = r'D:\01-Учеба Иннополис 2023\Домашки\LabCV(47)\DataSet\demo\*'
for i in range(3):
    s = 'Модель ' + str(i) + '\nКол-во эпох: ' + str(epochs_var[i]) + \
        '\nСкорость обучения: ' + str(lr_var[i]) + '\nСтепень усреднения: ' + str(momentum_var[i])
    st = time.time()
    optimizer = optim.SGD(model.parameters(), lr=lr_var[i], momentum=momentum_var[i])
    num_epochs = epochs_var[i]
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {running_loss / dataset_sizes['train']}")

    # Сохранение модели
    # torch.save(model.state_dict(), 'bfly_model.pth')
    s = s + '\nВремя обучения модели:' + str(time.time() - st)
    for img in glob.glob(path):
        predicted_class = classify_image(img)
        s = s + '\nФайл ' + os.path.split(img)[1] + ': ' + predicted_class +'\n'
    with open('log.txt', 'a') as file:
        file.write(s)
