import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets
from torchsummary import summary
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

# Set seed for reproducibility
torch.manual_seed(410)
np.random.seed(410)
random.seed(410)


class CustomFashionMNIST(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.labels = dataset.targets

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx_1):
        img1, label1 = self.dataset[idx_1]
        is_same_class = random.randint(0, 1)

        if is_same_class:
            idx_2 = np.where(self.labels == label1)[0]
        else:
            idx_2 = np.where(self.labels != label1)[0]

        img2, label2 = self.dataset[random.choice(idx_2)] 
        pair_label = torch.tensor(float(label1 != label2)) # 1 if the images are not similar, 0 otherwise

        return img1, img2, pair_label
    
    def visualize_imgs_pair(self, is_similar):
        # visualize a pair of images
        assert is_similar in [True, False], "is_similar should be either True or False"
        idx_1 = random.choice(range(len(self.dataset)))
        img1, label1 = self.dataset[idx_1]
        if is_similar:
            print("Images pair with the same class:")
            idx_2 = np.where(self.labels == label1)[0]
        else:
            print("Images pair with different classes:")
            idx_2 = np.where(self.labels != label1)[0]

        img2, label2 = self.dataset[random.choice(idx_2)]

        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(img1.squeeze(), cmap="gray")
        ax[1].imshow(img2.squeeze(), cmap="gray")
        ax[0].axis("off")
        ax[1].axis("off")
        ax[0].set_title(f"{self.dataset.classes[label1]}")
        ax[1].set_title(f"{self.dataset.classes[label2]}")
        plt.show()
        return (img1, img2), (label1, label2), (idx_1, idx_2)
    
class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Margin is used to compute the loss as well as to threshold the distance between the embeddings and compute accuracy.
    """
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def threshold(self, output1, output2):
        """
        distance > margin -> 1 else 0
        """
        distance = nn.functional.pairwise_distance(output1, output2, 2)
        threshold = distance.clone()
        threshold.data.fill_(self.margin)
        return (distance > self.margin).float().squeeze()
    
    def forward(self, output1, output2, label):
        distance = nn.functional.pairwise_distance(output1, output2, 2)
        loss = torch.mean((1 - label) * torch.pow(distance, 2) + (label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))
        return loss

    
class AbstractSiameseNet(nn.Module):
    def __init__(self,):
        super().__init__()
        self.contrastive_loss = False
    
class SiameseTrainer(object):
    def __init__(self, siamese_model, train_dataset, val_dataset, loss_function, batch_size, n_epochs, optimizer, path_prefix = ""):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = siamese_model
        self.loss_function = loss_function

        if self.loss_function.__class__.__name__ == "ContrastiveLoss":
            self.model.contrastive_loss = True
        else:
            self.model.contrastive_loss = False

        self.model.to(self.device)
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.n_epochs = n_epochs

        self.train_dataset = train_dataset
        self.val_datset = val_dataset
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_dataloader = torch.utils.data.DataLoader(self.val_datset, batch_size=self.batch_size, shuffle=False)

        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def train(self, epoch):
        self.model.train()
        train_loss = 0
        correct_pred = 0
        total_pred = 0
        for batch_idx, (img1, img2, label) in tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader)):
            img1, img2, label = img1.to(self.device), img2.to(self.device), label.to(self.device)
            
            self.optimizer.zero_grad()
            if self.model.contrastive_loss:
                out1, out2 = self.model(img1, img2)
                loss = self.loss_function(out1, out2, label)
                predicted = self.loss_function.threshold(out1, out2)  # Apply threshold
            else:
                out = self.model(img1, img2)
                loss = self.loss_function(out, label.view(-1, 1))
                predicted = (out < 0.5).float().squeeze() 

            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()

            correct_pred += (predicted == label).sum().item()
            total_pred += len(label)


        train_loss /= len(self.train_dataloader.dataset)/self.batch_size
        self.train_losses.append(train_loss)
        self.train_accuracies.append(correct_pred/total_pred)
        if epoch % 5 == 0:
            print(f"Epoch [{epoch}], train_loss: {train_loss:.4f}, train_acc: {correct_pred/total_pred:.4f}", end="")
    
    def validate(self, epoch):
        self.model.eval()
        val_loss = 0
        correct_pred = 0
        total_pred = 0
        with torch.no_grad():
            for img1, img2, label in self.val_dataloader:
                img1, img2, label = img1.to(self.device), img2.to(self.device), label.to(self.device)
                if self.model.contrastive_loss:
                    out1, out2 = self.model(img1, img2)
                    val_loss += self.loss_function(out1, out2, label).item()
                    predicted = self.loss_function.threshold(out1, out2)
                else:
                    out = self.model(img1, img2)
                    val_loss += self.loss_function(out, label.view(-1, 1)).item()
                    predicted = (out < 0.5).float().squeeze()

                correct_pred += (predicted == label).sum().item()
                total_pred += len(label)

        val_loss /= len(self.val_dataloader.dataset)/self.batch_size
        self.val_losses.append(val_loss)
        self.val_accuracies.append(correct_pred/total_pred)
        if epoch % 5 == 0:
            print(f", val_loss: {val_loss:.4f}, val_acc: {correct_pred/total_pred:.4f}")
    
    def train_loop(self, validation=True):
        for epoch in range(1, self.n_epochs+1):
            self.train(epoch)
            if validation:
                self.validate(epoch)

        return self.train_losses, self.val_losses, self.train_accuracies, self.val_accuracies
    
    def plot_losses(self, train_losses, val_losses):
        plt.plot(train_losses, label="Training Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.xticks(np.arange(0, self.n_epochs, 5), rotation=90)
        plt.ylabel("Loss")
        plt.title(f"Optimizer: {self.optimizer.__class__.__name__}, Loss: {self.loss_function.__class__.__name__}, BS: {self.batch_size}, LR: {self.optimizer.param_groups[0]['lr']}")
        plt.legend()
        plt.show()

    def plot_accuracies(self, train_accuracies, val_accuracies):
        plt.plot(train_accuracies, label="Training Accuracy")
        plt.plot(val_accuracies, label="Validation Accuracy")
        plt.xlabel("Epochs")
        plt.xticks(np.arange(0, self.n_epochs, 5), rotation=90)
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        plt.title(f"Optimizer: {self.optimizer.__class__.__name__}, Loss: {self.loss_function.__class__.__name__}, BS: {self.batch_size}, LR: {self.optimizer.param_groups[0]['lr']}")
        plt.legend()
        plt.show()

class ClassifierTrainer(object):
    def __init__(self, model, train_dataset, val_dataset, loss_function, batch_size, n_epochs, optimizer, path_prefix = ""):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = model
        self.loss_function = loss_function
        self.model.to(self.device)
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.n_epochs = n_epochs

        self.train_dataset = train_dataset
        self.val_datset = val_dataset
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_dataloader = torch.utils.data.DataLoader(self.val_datset, batch_size=self.batch_size, shuffle=False)

        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def train(self, epoch):
        self.model.train()
        train_loss = 0
        correct_pred = 0
        total_pred = 0
        for batch_idx, (img, label) in tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader)):
            img, label = img.to(self.device), label.to(self.device)
            
            self.optimizer.zero_grad()
            out = self.model(img)

            # compute accuracy
            predicted = (out.argmax(dim=1) == label).float().sum()
            correct_pred += predicted.item()
            total_pred += len(label)
    
            label = F.one_hot(label, num_classes=10).float()
            loss = self.loss_function(out, label)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()


        train_loss /= len(self.train_dataloader.dataset)/self.batch_size
        self.train_losses.append(train_loss)
        self.train_accuracies.append(correct_pred/total_pred)
        if epoch % 5 == 0:
            print(f"Epoch [{epoch}], train_loss: {train_loss:.4f}, train_acc: {correct_pred/total_pred:.4f}", end="")
    
    def validate(self, epoch):
        self.model.eval()
        val_loss = 0
        correct_pred = 0
        total_pred = 0
        with torch.no_grad():
            for img, label in self.val_dataloader:
                img, label = img.to(self.device), label.to(self.device)
                out = self.model(img)

                predicted = (out.argmax(dim=1) == label).float().sum()
                correct_pred += predicted.item()
                total_pred += len(label)

                label = F.one_hot(label, num_classes=10).float()
                val_loss += self.loss_function(out, label).item()

        val_loss /= len(self.val_dataloader.dataset)/self.batch_size
        self.val_losses.append(val_loss)
        self.val_accuracies.append(correct_pred/total_pred)
        if epoch % 5 == 0:
            print(f", val_loss: {val_loss:.4f}, val_acc: {correct_pred/total_pred:.4f}")

    def train_loop(self, validation=True):
        for epoch in range(1, self.n_epochs+1):
            self.train(epoch)
            if validation:
                self.validate(epoch)

        return self.train_losses, self.val_losses, self.train_accuracies, self.val_accuracies
    
    def predict(self, img):
        self.model.eval()
        with torch.no_grad():
            img = img.to(self.device)
            out = self.model(img)
            return out.argmax(dim=1).item()
    
    def plot_losses(self, train_losses, val_losses):
        plt.plot(train_losses, label="Training Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.xticks(np.arange(0, self.n_epochs, 5), rotation=90)
        plt.ylabel("Loss")
        plt.title(f"Optimizer: {self.optimizer.__class__.__name__}, Loss: {self.loss_function.__class__.__name__}, BS: {self.batch_size}, LR: {self.optimizer.param_groups[0]['lr']}")
        plt.legend()
        plt.show()

    def plot_accuracies(self, train_accuracies, val_accuracies):
        plt.plot(train_accuracies, label="Training Accuracy")
        plt.plot(val_accuracies, label="Validation Accuracy")
        plt.xlabel("Epochs")
        plt.xticks(np.arange(0, self.n_epochs, 5), rotation=90)
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        plt.title(f"Optimizer: {self.optimizer.__class__.__name__}, Loss: {self.loss_function.__class__.__name__}, BS: {self.batch_size}, LR: {self.optimizer.param_groups[0]['lr']}")
        plt.legend()
        plt.show()

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.xticks(ticks=np.arange(10) + 0.5, labels=classes, rotation=90)
    plt.yticks(ticks=np.arange(10) + 0.5, labels=classes, rotation=0)
    plt.show()

def visualize_classes(fashion_mnist_dataset):
    # visualize a random image from each class
    fig, ax = plt.subplots(2, 5, figsize=(15, 7))
    for i in range(10):
        idx = np.where(fashion_mnist_dataset.targets == i)[0][0]
        img, label = fashion_mnist_dataset[idx]
        ax[i//5, i%5].imshow(img.squeeze(), cmap="gray")
        ax[i//5, i%5].axis("off")
        ax[i//5, i%5].set_title(fashion_mnist_dataset.classes[label])
    plt.show()
