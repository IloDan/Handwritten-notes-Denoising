from src.dataset import train_dataloader, test_dataloader
from src.model import UNet
from src.config import DEVICE, IN_CHANNELS, OUT_CHANNELS, NUM_EPOCHS, LEARNING_RATE
import torch
import torch.nn as nn
from torch.nn import DataParallel
from tqdm import tqdm
import clearml
from torchvision.transforms import ToPILImage
import random


torch.cuda.empty_cache()
# Inizializza il Task di ClearML
task = clearml.Task.init(project_name='Unet_Danilo_parallel_001', task_name='Training')



# Update the Task with new parameters or any other relevant information
task.set_parameters({
    "in_channels": IN_CHANNELS,
    "out_channels": OUT_CHANNELS,
    "num_epochs": NUM_EPOCHS,
    "learning_rate": LEARNING_RATE
})


net = UNet(IN_CHANNELS, OUT_CHANNELS).to(DEVICE)
#net.load_state_dict(torch.load('/homes/dcaputo/handwritten-math-recognition/UNET/trained/Unet_1024_2epochs.pth'))
net = net.to(DEVICE)

opt = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
#scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LEARNING_RATE, steps_per_epoch=len(train_dataloader), epochs=NUM_EPOCHS)
criterion = nn.MSELoss()
loss_train = []
loss_test  = []
# Precompute random indices for a fixed number of epochs
total_images = len(test_dataloader.dataset)
all_indices = list(range(total_images))
random_indices_per_epoch = [random.sample(all_indices, 5) for _ in range(NUM_EPOCHS)]
for e in range(NUM_EPOCHS):
    pbar = tqdm(total=len(train_dataloader), desc=f'Epoch {e+1} - 0%', dynamic_ncols=True)
    
    total_loss = 0.0
    num_batches = 0
    net.train()
    for i, (x, y) in enumerate(train_dataloader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        y_pred = net(x)
        loss = criterion(y_pred, y)
        loss.backward()
        opt.step()
        pbar.update(1)
        pbar.set_description(f'Epoch {e+1} - {round(i / len(train_dataloader) * 100)}% -- loss {loss.item():.2f}')
        total_loss += loss.item()
        num_batches += 1

    pbar.close()
    avg_loss = total_loss / num_batches
    loss_train.append(avg_loss)
    print(f"Loss on train for epoch {e+1}: {loss_train[e]}")
    task.get_logger().report_scalar(title='Loss', series='Train_loss', value=loss_train[e], iteration=e+1)
    

    mse_temp = 0
    cont = 0
    net.eval()
    
   
    random_indices = random_indices_per_epoch[e]
    with torch.no_grad():
        for c, (x, y) in enumerate(test_dataloader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_pred = net(x)
            mse_temp += criterion(y_pred, y).cpu().item()
            cont += 1
        for idx in random_indices:
            x, _ = test_dataloader.dataset[idx]  # Load the image by index
            x = x.unsqueeze(0).to(DEVICE)  # Add batch dimension
            y_pred = net(x)
            single_output_image = (y_pred * 0.1477) + 0.9335
            output_pil = ToPILImage()(single_output_image.squeeze(0).cpu())
            output_pil.save('test/' + str(e + 1) + '_output_' + str(idx) + '.png')

    loss_test.append(mse_temp/cont)
    print(f"Loss on test for epoch {e+1}: {loss_test[e]}")
    task.get_logger().report_scalar(title='Loss', series='Test_loss', value=loss_test[e], iteration=e+1)
    torch.save(net.state_dict(), f'Unet_1024_{e+1}epochs001.pth')
    print(f"Model saved at epoch {e+1}")
    task.upload_artifact(f'Unet_1024_{e+1}epochs001.pth', artifact_object=f'Unet_1024_{e+1}epochs001.pth')

# Completa il Task di ClearML
task.close()