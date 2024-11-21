import os
import time
import wandb
import torch
import argparse
from tqdm import tqdm
from PIL import Image
from CLIP import clip
from CLIP.clip.clip import _transform
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn


def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 

class image_title_dataset(Dataset):
    def __init__(self, dataset, preprocess, base_text):
        self.dataset = dataset
        self.preprocess = preprocess
        self.base_text = base_text

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        img = self.preprocess(x)
        label = clip.tokenize(self.base_text+f"{self.dataset.classes[y]}")
        return img, label.squeeze()

def finetune(model, train_dataloader, optimizer, device):
    
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    
    for images, texts in tqdm(train_dataloader, desc="Training"):
        images = images.to(device)
        texts = texts.to(device)
        
        # Get image features
        logits_per_image, logits_per_text = model(images, texts)
        ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
        # print("logits_shape", logits_per_image.shape, logits_per_text.shape)
        total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text, ground_truth))/2

        wandb.log({"loss": total_loss.item()})
        optimizer.zero_grad()
        total_loss.backward()
        convert_models_to_fp32(model)
        optimizer.step()


    return model

def zero_shot_accuracy(model, test_dataloader, text_features, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_dataloader, desc="Evaluating"):
            images = images.to(torch.float32).to(device)
            labels = labels.to(torch.float32).to(device)
            
            # Get image features
            image_features = model.encode_image(images)
            # image_features = F.normalize(image_features, dim=1)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # Convert both to float32
            image_features = image_features.float()
            text_features = text_features.float()
            
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            
            # Get predictions
            _, predictions = similarity.topk(1)
            correct += (predictions.squeeze() == labels).sum().item()
            total += labels.size(0)
            
    return 100 * correct / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--model_name", type=str, default="ViT-B/32")
    parser.add_argument("--model_path", type=str, default="./model_path")
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--save_path", type=str, default="./save_path")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize wandb
    wandb.init(project="CLIP-finetune", config=args)
    
    # Load CLIP model
    model, preprocess = clip.load(args.model_name, download_root=args.model_path, device=device)
    
    # Load CIFAR-100 dataset
    train_dataset = CIFAR100(root=args.data_path, train=True, download=True)
    test_dataset = CIFAR100(root=args.data_path, train=False, download=True,
                           transform=_transform(model.visual.input_resolution))
    
    finetune_dataset = image_title_dataset(train_dataset, preprocess, "a photo of a ")
    train_dataloader = DataLoader(finetune_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create text embeddings for each class
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in train_dataset.classes]).to(device)
    
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features = F.normalize(text_features, dim=1)
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)

    orig_type = dict()
    for name, param in model.named_parameters():
        orig_type[name] = param.dtype
    
    # Before training loop
    best_accuracy = 0
    os.makedirs(args.save_path, exist_ok=True)
    
    # Training loop
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        # Train
        model.train()
        model = finetune(model, train_dataloader, optimizer, device)

        # Evaluate
        model.eval()
        accuracy = zero_shot_accuracy(model, test_dataloader, text_features, device)
        print(f"Zero-shot accuracy: {accuracy:.2f}%")
        
        _is_save = 0
        # Save best model
        if accuracy > best_accuracy:
            _is_save = 1
            best_accuracy = accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': accuracy,
            }, os.path.join(args.save_path, 'fientunebest_model.pt'))
            print(f"New best model saved with accuracy: {accuracy:.2f}%")
        
        wandb.log({"accuracy": accuracy, "is_save": _is_save})

    print(f"Training finished. Best accuracy: {best_accuracy:.2f}%")
    wandb.finish()



if __name__ == "__main__":
    main()
