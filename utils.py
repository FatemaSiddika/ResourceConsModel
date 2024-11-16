import argparse
import torch
import torch.optim as optim

def get_args():
    parser = argparse.ArgumentParser(description='CIFAR-10 Training')
    parser.add_argument('--model', type=str, default='mobilenetv2', choices=['mobilenetv2', 'resnet34', 'vit_base'])
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--IsPreTrained', type=int, default=0)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--lr', type=float, default=None)  
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=None) 
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()

def get_optimizer(model, args):
    if args.lr is None or args.weight_decay is None:
        if args.model == 'resnet34':
            args.lr = 0.1  
            args.weight_decay = 5e-4 
        elif args.model == 'mobilenetv2':
            args.lr = 0.05  
            args.weight_decay = 4e-5  
        elif args.model == 'vit_base':
            args.lr = 3e-3  
            args.weight_decay = 1e-2
        else:
            raise ValueError(f"Unsupported model: {args.model}")
    
    if args.model in ['resnet34', 'mobilenetv2']:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    elif args.model == 'vit_base':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    return optimizer, scheduler

def print_info(args, train_dataset, test_dataset):
    print(f"Training {args.model} on CIFAR-10")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr if args.lr is not None else 'Default'}")
    print(f"Momentum: {args.momentum}")
    print(f"Weight decay: {args.weight_decay if args.weight_decay is not None else 'Default'}")
    print(f"Epochs: {args.epochs}")
    print(f"Device: {args.device}")
    print()