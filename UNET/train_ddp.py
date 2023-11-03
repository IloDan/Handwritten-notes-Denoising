from src.dataset import train_dataloader, test_dataloader
from src.model import UNet
from src.config import IN_CHANNELS, OUT_CHANNELS, NUM_EPOCHS, LEARNING_RATE, MEAN, STD
import torch
import torch.nn as nn
from tqdm import tqdm
import clearml
# from torchvision.transforms import ToPILImage
# import random
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import os
import glob
import time
from collections import deque

torch.cuda.empty_cache()

# Inizializza il processo di DistributedDataParallel (DDP)
dist.init_process_group(backend='nccl')

# Inizializza il Task principale di ClearML
task_main = clearml.Task.init(project_name='Unet_Danilo_ddp_finale', task_name='Training')

# Update the Task with new parameters or any other relevant information
task_main.set_parameters({
    "in_channels": IN_CHANNELS,
    "out_channels": OUT_CHANNELS,
    "num_epochs": NUM_EPOCHS,
    "learning_rate": LEARNING_RATE
})

rank = dist.get_rank()
device_id = rank


# Specifica la cartella dei modelli
model_folder = "/work/cvcs_2023_group11/checkpoints/"

# Trova il percorso del modello più recente nella cartella dei modelli
try:
    latest_model_path = max(glob.glob(os.path.join(model_folder, "*.pth")), key=os.path.getctime)
except ValueError:
    # Nessun file .pth trovato nella cartella, crea un nuovo modello
    latest_model_path = None

# Controlla se il modello più recente esiste
if latest_model_path:
    # Crea il modello UNet e spostalo sul dispositivo specificato (device_id)
    net = UNet(IN_CHANNELS, OUT_CHANNELS).to(device_id)
    # Carica lo stato del modello più recente
    net.load_state_dict(torch.load(latest_model_path))
    # Incapsula il modello in DistributedDataParallel (DDP) per l'addestramento parallelo
    net = DDP(net)
    print("Latest model loaded from:", latest_model_path)
else:
    # Crea un nuovo modello UNet e incapsulalo in DistributedDataParallel (DDP)
    net = UNet(IN_CHANNELS, OUT_CHANNELS).to(device_id)
    net = DDP(net)
    print("Creating a new model")


opt = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    opt, max_lr=0.03, steps_per_epoch=len(train_dataloader),
    epochs=NUM_EPOCHS, cycle_momentum=True
)
criterion = nn.MSELoss()
loss_train = []
loss_test = []

# # Precompute random indices for a fixed number of epochs
# total_images = len(test_dataloader.dataset)
# all_indices = list(range(total_images))
# random_indices_per_epoch = [random.sample(all_indices, 5) for _ in range(NUM_EPOCHS)]


# Variabili per il controllo dei checkpoint
checkpoint_interval = 3600  # Un'ora in secondi
max_checkpoints = 30
last_checkpoint_time = time.time()
# Utilizza una coda per mantenere i nomi dei checkpoint salvati
saved_checkpoints = deque(maxlen=max_checkpoints)

# Cartella per i checkpoint
os.makedirs("/work/cvcs_2023_group11/checkpoints", exist_ok=True)

for e in range(NUM_EPOCHS):
    pbar = tqdm(total=len(train_dataloader), 
                desc=f'Epoch {e + 1} - 0%', 
                dynamic_ncols=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataloader.dataset, 
        num_replicas=dist.get_world_size(), 
        rank=rank, 
        shuffle=True
    )
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataloader.dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=False
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataloader.dataset,
        batch_size=train_dataloader.batch_size,
        sampler=train_sampler,
        num_workers=train_dataloader.num_workers,
        pin_memory=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataloader.dataset,
        batch_size=test_dataloader.batch_size,
        sampler=test_sampler,
        num_workers=test_dataloader.num_workers,
        pin_memory=True
    )

    train_sampler.set_epoch(e)  # Set epoch for distributed sampler
    total_loss = 0.0
    num_batches = 0
    net.train()

    for i, (x, y) in enumerate(train_dataloader):
        x, y = x.to(device_id), y.to(device_id)
        opt.zero_grad()
        y_pred = net(x)
        loss = criterion(y_pred, y)
        loss.backward()
        opt.step()
        scheduler.step()
        pbar.update(1)
        pbar.set_description(f'Epoch {e + 1} - {round(i / len(train_dataloader) * 100)}% -- loss {loss.item():.2f}')
        total_loss += loss.item()
        num_batches += 1

        # Salvataggio del modello solo sul processo principale (rank 0)
        if rank == 0:
            current_time = time.time()
            if current_time - last_checkpoint_time >= checkpoint_interval and len(saved_checkpoints) < max_checkpoints:
                # Salva il checkpoint attuale
                checkpoint_name = f'Unet_1024_{e+1}epochs_check_{current_time:.0f}.pth'
                checkpoint_path = os.path.join('/work/cvcs_2023_group11/checkpoints', checkpoint_name)  # Percorso completo al checkpoint
                torch.save(net.module.state_dict(), checkpoint_path)
                print(f"Model saved at epoch {e+1} - Hourly checkpoint")
                saved_checkpoints.append(checkpoint_path)
                last_checkpoint_time = current_time
                task_main.upload_artifact(checkpoint_path, artifact_object=checkpoint_name)
        

    pbar.close()
    avg_loss = total_loss / num_batches
    loss_train.append(avg_loss)
    print(f"Loss on train for epoch {e + 1}: {loss_train[e]}")
    task_main.get_logger().report_scalar(title='Loss', series='Train_loss', value=loss_train[e], iteration=e + 1)

    mse_temp = 0
    cont = 0
    net.eval()

    # random_indices = random_indices_per_epoch[e]
    with torch.no_grad():
        for c, (x, y) in enumerate(test_dataloader):
            x, y = x.to(device_id), y.to(device_id)
            y_pred = net(x)
            mse_temp += criterion(y_pred, y).cpu().item()
            cont += 1
        # for idx in random_indices:
        #     x, _ = test_dataloader.dataset[idx]  # Load the image by index
        #     x = x.unsqueeze(0).to(device_id)  # Add batch dimension
        #     y_pred = net(x)
        #     single_output_image = (y_pred * STD) + MEAN
        #     output_pil = ToPILImage()(single_output_image.squeeze(0).cpu())
        #     output_pil.save('test/' + str(e + 1) + '_output_' + str(idx) + '.png')

    loss_test.append(mse_temp / cont)
    print(f"Loss on test for epoch {e + 1}: {loss_test[e]}")
    task_main.get_logger().report_scalar(title='Loss', series='Test_loss', value=loss_test[e], iteration=e + 1)

    # Salvataggio del modello solo sul processo principale (rank 0)
    if rank == 0:
        torch.save(net.module.state_dict(), f'/work/cvcs_2023_group11/checkpoints//Unet_1024_{e+1}epochsADAM_{current_time:.0f}.pth')
        print(f"Model saved at epoch {e+1}")
        task_main.upload_artifact(f'Unet_1024_{e+1}epochsADAM_{current_time:.0f}.pth', artifact_object=f'Unet_1024_{e+1}epochsADAM_{current_time:.0f}.pth')

# Completa il Task principale di ClearML
task_main.close()
