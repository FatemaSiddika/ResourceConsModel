import torch
from models import get_model
from data import get_dataloaders
from train import train, test
from utils import get_args, print_info, get_optimizer
from tqdm import trange

def main():
    args = get_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    train_loader, test_loader = get_dataloaders(args.batch_size)
    model = get_model(args.model, args.IsPreTrained, args.num_classes).to(args.device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer, scheduler = get_optimizer(model, args)
    
    print_info(args, train_loader.dataset, test_loader.dataset)
    
    for epoch in trange(args.epochs, desc="Epochs"):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, args.device)
        test_loss, test_acc = test(model, test_loader, criterion, args.device)
        scheduler.step()
        
        print(f"Epoch [{epoch+1}/{args.epochs}]:")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        print()

if __name__ == '__main__':
    main()