import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch import nn
from tqdm import tqdm
from colors import color
from models.CNNModelV1 import CNNModelV1
from models.CNNModelV2 import CNNModelV2
from models.CNNModelV3 import CNNModelV3
from data.datasets import train_data
from data.datasets import val_data
import os


# Model initialisation
cnn_v1 = CNNModelV1(input_shape=3, hidden_units=4, output_shape=train_data.genre_count)
cnn_v2 = CNNModelV2(input_shape=3, hidden_units=8, output_shape=train_data.genre_count)
cnn_v3 = CNNModelV3(input_shape=3, hidden_units=4, output_shape=train_data.genre_count)


model_options = ['cnn_v1', 'cnn_v2', 'cnn_v3']

device_options = ['cpu', 'cuda', 'ipu', 'xpu', 'mkldnn', 
                  'opengl', 'opencl', 'ideep', 'hip', 've', 
                  'fpga', 'ort', 'xla', 'lazy', 'vulkan', 'mps', 
                  'meta', 'hpu', 'mtia']


def val_accuracy(y_pred, y_true):
    correct_label = torch.eq(y_true, y_pred).sum().item()
    acc = (correct_label / len(y_pred)) * 100
    return acc


def train_model(model, batch_size: int, learn_rate:float, device=device_options):
    # Dataloader
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=learn_rate)
    
    if model == cnn_v2:
        scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=5)
    else:
        scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=1.0, total_iters=1)
    
    # Load existing model to avoid training entirely new model
    if os.path.isfile(f'./models/saved/{model.__class__.__name__}.pth'):
        checkpoint = torch.load(f'./models/saved/{model.__class__.__name__}.pth')
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        best_acc = checkpoint['best_acc']
        lowest_val_loss = checkpoint['lowest_val_loss']
        lowest_train_loss = checkpoint['lowest_train_loss']
        best_epoch = checkpoint['best_epoch']
        
    #model = torch.compile(model.to(device))
    model = model.to(device)
    lowest_val_loss = 100.0
    lowest_train_loss = 100.0
    best_acc = 0.0
    best_epoch = 0
    epochs = 50
    print(f'\nTraining {model.__class__.__name__}...')
    for epoch in range(epochs):
        print(f'Epoch {epoch+1} / {epochs}')
        model.train()
        train_loss = 0
        batch_progress = tqdm(total=len(train_dataloader), desc='Batch Progress', leave=False)
        for img, label in train_dataloader:
            # 1. Forward pass
            label_pred = model(img.to(device))
            # 2. Calculate loss (per batch)
            loss = loss_fn(label_pred, label.to(device))
            train_loss += loss
            # 3. Optimizer zero grad
            optimizer.zero_grad()
            # 4. Loss backward
            loss.backward()
            # 5. Optimizer step
            optimizer.step()
            batch_progress.update()
        batch_progress.close()
        scheduler.step()
        
        # Divide total train loss by length of train loader
        train_loss /= len(train_dataloader)
        
        # Testing with validation set
        val_loss = 0
        val_acc = 0
        model.eval()
        with torch.inference_mode():
           for img, label in val_dataloader:
               val_pred = model(img.to(device))
               val_loss += loss_fn(val_pred, label.to(device))
               val_acc += val_accuracy(y_pred=val_pred.argmax(dim=1), y_true=label.to(device))
           
           # Total test loss divided by length of val dataloader
           val_loss /= len(val_dataloader)
           # Total accuracy divided by length of val dataloader
           val_acc /= len(val_dataloader)
        
        print(f'\rVal loss: {val_loss}')
        print(f'Val acc: {val_acc} %')
        print(f'Train loss: {train_loss}')
        
        # Save if accuracy is better and validation loss is lower than previously
        if val_acc >= best_acc and train_loss <= lowest_train_loss:
            best_acc = val_acc
            lowest_val_loss = val_loss
            lowest_train_loss = train_loss
            best_epoch = epoch + 1
            state = {'epoch': epoch+1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'scheduler': scheduler.state_dict(), 'loss': loss, 'best_acc': best_acc,
                     'lowest_val_loss': lowest_val_loss, 'lowest_train_loss': lowest_train_loss,
                     'best_epoch': best_epoch}
            
            torch.save(state, f'./models/saved/{model.__class__.__name__}.pth')
            print(f'{color.GREEN}Saving at Epoch {epoch + 1}{color.END}')
        
        print('______________________________________') 
        print('Saved State Info') 
        print(f' > Train loss: {lowest_train_loss}')
        print(f' > Val acc: {best_acc} %')
        print(f' > Epoch: {best_epoch}\n')
    print(f'\n{color.GREEN}Trained {model.__class__.__name__} successfully!{color.END}\n')
    
