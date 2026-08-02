from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

transform = transforms.ToTensor()

train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

#dir(test_dataset) # just to check the methods and attributes inside of the object

#to make the data being delivered in batches 
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

images, labels = next(iter(train_loader)) # returns image - label for all the data one by one

print(type(images))
print(images.shape)

print(type(labels))
print(labels.shape)

classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]
import matplotlib.pyplot as plt
image = images[0]
label = labels[0]

print(image.shape)
print(label)
print(classes[label])

image = image.permute(1, 2, 0) # matplt expected format erxpects a format of 

plt.imshow(image)
plt.show()

