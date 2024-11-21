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
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.nn as nn

def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        if p.grad is not None:
            p.grad.data = p.grad.data.float() 

def get_logits(image_features, text_features, logit_scale):
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()
    return logits_per_image, logits_per_text

class image_title_dataset(Dataset):
    def __init__(self, dataset, base_text):
        self.dataset = dataset
        self.base_text = base_text
        if hasattr(dataset, 'dataset'):
            self.classes = dataset.dataset.classes
        else:
            self.classes = dataset.classes

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        label = clip.tokenize(self.base_text+f"{self.classes[y]}")
        return x, y, label.squeeze()

def contrastive_unlearn(args,
                        model,
                        retain_dataloader,
                        unlearn_dataloader,
                        optimizer,
                        device):
    
    for ul_images, ul_labels in tqdm(unlearn_dataloader, desc="Unlearning"):
        _retain_iter = iter(retain_dataloader)
        for _ in range(args.num_sampling):
            try:
                r_images, r_labels, r_texts = next(_retain_iter)
            except StopIteration:
                _retain_iter = iter(retain_dataloader)
                r_images, r_labels, r_texts = next(_retain_iter)
            
            _ul_images = ul_images.clone().detach().to(device)
            _ul_labels = ul_labels.clone().detach().to(device)
            _r_images = r_images.to(device)
            _r_labels = r_labels.to(device)
            _r_texts = r_texts.to(device)

            _images = torch.cat([_ul_images, _r_images])

            image_features = model.encode_image(_images)
            image_features = F.normalize(image_features, dim=1)

            r_text_features = model.encode_text(_r_texts)
            r_text_features = F.normalize(r_text_features, dim=1)
            
            ul_image_features = image_features[:len(_ul_images)]
            r_image_features = image_features[len(_ul_images):]

            ul_loss = unlearn_loss(ul_image_features,
                                    r_image_features,
                                    _ul_labels,
                                    _r_labels,
                                    args.temperature,
                                    device,
                                    single_class=True)

            loss = args.loss_ratio *ul_loss  
            optimizer.zero_grad()
            loss.backward()
            convert_models_to_fp32(model)
            optimizer.step()

    return model

def finetune(model, train_dataloader, optimizer, device):
    
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    
    for images, _, texts in tqdm(train_dataloader, desc="Training"):
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

def retain_loss(r_image_logits, r_text_logits, device):
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    ground_truth = torch.arange(len(r_image_logits), dtype=torch.long, device=device)
    loss_img = loss_img(r_image_logits, ground_truth)
    loss_txt = loss_txt(r_text_logits, ground_truth)
    loss = loss_img + loss_txt
    return loss

def self_reconstruction_loss(r_image_features,
                            r_labels,
                            temperature,
                            base_temperature=0.07):
    # Reconstruct the image features of the retain dataset.
    return reconstruction_loss(r_image_features, r_image_features, r_labels, r_labels, temperature, base_temperature)

def reconstruction_loss(r_image_features_1,
                        r_image_features_2,
                        r_labels_1,
                        r_labels_2,
                        temperature,
                        device,
                        base_temperature=0.07):

    r_labels_1 = r_labels_1.contiguous().view(-1, 1)
    r_labels_2 = r_labels_2.contiguous().view(-1, 1)

    mask = torch.eq(r_labels_1, r_labels_2.T)
    p_mask = (mask).clone().float().to(device)
    n_mask = (~mask).clone().float().to(device)

    orig_logits = torch.matmul(r_image_features_1, r_image_features_2.T)
    logits_max, _ = torch.max(orig_logits, dim=1, keepdim=True)
    logits = orig_logits - logits_max.detach()

    exp_logits = torch.exp(logits) * n_mask + 1e-20
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    mean_log_prob = (p_mask * log_prob).sum(1) / p_mask.sum(1)

    loss = -(temperature / base_temperature) * mean_log_prob
    loss = loss.mean()

    return loss


def unlearn_loss(ul_image_features,
                 r_image_features,
                 ul_labels,
                 r_labels,
                 temperature,
                 device,
                 base_temperature=0.07,
                 single_class=False):
    
    unlearn_label = ul_labels.contiguous().view(-1, 1)
    retain_label = r_labels.contiguous().view(-1, 1)
    
    mask = torch.eq(unlearn_label, retain_label.T)
    p_mask = (~mask).clone().float().to(device)
    p_count = torch.sum(mask).item()

    n_mask = mask.clone().float().to(device)
    n_count = torch.sum(n_mask).item()
    n_add_mask = (~(n_mask.sum(1).bool())).int().contiguous().view(-1, 1).to(device)

    # print(len(ul_labels), len(r_labels))
    # print(p_mask.shape, n_mask.shape)

    orig_logits = torch.matmul(ul_image_features, r_image_features.T)
    print(ul_image_features.shape, r_image_features.shape)
    print(orig_logits.shape)

    logits = torch.div(orig_logits, temperature)
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()

    exp_logits = torch.exp(logits) + 1e-20
    p_logits = exp_logits * p_mask
    n_logits = (exp_logits * n_mask).sum(1, keepdim=True)

    if single_class or (n_mask.sum(1) == 0).any():
        # Unlearning for single class or when a batch does not contain no negative sample for a class.
        n_logits += n_add_mask

    log_prob = p_logits - torch.log(n_logits)
    mean_log_prob = log_prob.sum(1) / p_mask.sum(1)

    loss = -(temperature / base_temperature) * mean_log_prob
    loss = loss.mean()

    return loss


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unlearn_epochs", type=int, default=10)
    parser.add_argument("--num_sampling", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--unlearn_class", type=int, default=0, choices=range(100))
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--model_name", type=str, default="ViT-B/32")
    parser.add_argument("--model_path", type=str, default="./model_path")
    parser.add_argument("--load_path", type=str, default="")
    parser.add_argument("--save_path", type=str, default="./save_path")
    parser.add_argument("--loss_ratio", type=float, default=0.5)
    parser.add_argument("--wandb_project", type=str, default="CLIP-unlearn")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--unlearn_lr", type=float, default=1e-12)

    return parser


def get_dataset(args, dataset):
    # Split dataset into unlearn and retain datasets based on class
    unlearn_indices = [i for i, (_, label) in enumerate(dataset) if label == args.unlearn_class]
    retain_indices = [i for i, (_, label) in enumerate(dataset) if label != args.unlearn_class]

    # Create subset datasets
    unlearn_dataset = torch.utils.data.Subset(dataset, unlearn_indices)
    retain_dataset = torch.utils.data.Subset(dataset, retain_indices)

    return retain_dataset, unlearn_dataset


def zero_shot_accuracy(model, dataloader, text_features, device):
    all_predictions = []
    all_labels = []

    # Convert text_features to float32 to match image_features
    text_features = text_features.to(torch.float32)

    for images, labels in tqdm(dataloader, desc="Evaluating zero-shot accuracy"):
        # Move batch to device and ensure float32 type
        images = images.to(torch.float32).to(device)
        labels = labels.to(device)  # Remove unnecessary float32 conversion for labels
            
        # Get image features
        image_features = model.encode_image(images)
        image_features = image_features.to(torch.float32)  # Ensure float32
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        # Calculate similarities
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            
        # Get top predictions
        _, predictions = similarity.topk(1)
            
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())  # Add cpu() before numpy()
    
    # Calculate accuracy
    correct = sum(pred == label for pred, label in zip(all_predictions, all_labels))
    total = len(all_labels)
    accuracy = 100 * correct / total

    return accuracy

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    _run_name = "CIFAR-100" + "_" \
                + time.strftime("%Y-%m-%d_%H-%M-%S")

    wandb.init(project=args.wandb_project,
               name=_run_name,
               config=args
    )

    model, preprocess = clip.load(args.model_name, download_root=args.model_path)
    model.to(device)

    if len(args.load_path) > 0:
        checkpoint = torch.load(args.load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model checkpoint from {args.load_path}")


    train_dataset = CIFAR100(root=args.data_path, train=True, download=True, transform=_transform(model.visual.input_resolution))
    test_dataset = CIFAR100(root=args.data_path, train=False, download=True, transform=_transform(model.visual.input_resolution))

    print(type(train_dataset[0][0]))

    retain_dataset, unlearn_dataset = get_dataset(args, train_dataset)
    retain_test_dataset, unlearn_test_dataset = get_dataset(args, test_dataset)
    retain_dataset = image_title_dataset(retain_dataset, "a photo of a ")
    retain_sampler = torch.utils.data.RandomSampler(retain_dataset, replacement=True, num_samples=len(retain_dataset))
    retain_dataloader = DataLoader(retain_dataset, batch_size=args.batch_size, sampler=retain_sampler)
    unlearn_dataloader = DataLoader(unlearn_dataset, batch_size=args.batch_size, shuffle=True)

    retain_test_loader = DataLoader(retain_test_dataset, batch_size=args.batch_size, shuffle=False)
    unlearn_test_loader = DataLoader(unlearn_test_dataset, batch_size=args.batch_size, shuffle=False)

    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in train_dataset.classes]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)

    ul_optimizer = torch.optim.Adam(model.parameters(), lr=args.unlearn_lr,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)
    rt_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)
    
    with torch.no_grad():
        retain_test_acc = zero_shot_accuracy(model, retain_test_loader, text_features, device)
        unlearn_test_acc = zero_shot_accuracy(model, unlearn_test_loader, text_features, device)
        wandb.log({"retain_test_acc": retain_test_acc, "unlearn_test_acc": unlearn_test_acc})

    for idx in range(args.unlearn_epochs):
        print(f"Unlearning epoch {idx+1}/{args.unlearn_epochs}")
        model.train()
        model = finetune(model, retain_dataloader, rt_optimizer, device)
        model = contrastive_unlearn(args, model, retain_dataloader, unlearn_dataloader, ul_optimizer, device)
        model = model.float()
        with torch.no_grad():
            retain_test_acc = zero_shot_accuracy(model, retain_test_loader, text_features, device)
            unlearn_test_acc = zero_shot_accuracy(model, unlearn_test_loader, text_features, device)
            wandb.log({"retain_test_acc": retain_test_acc, "unlearn_test_acc": unlearn_test_acc})


    wandb.finish()

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)