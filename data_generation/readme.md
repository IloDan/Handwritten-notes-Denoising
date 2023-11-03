# Dataset Generation

- se si lavora su server unimore:
- Tutti i dati devono essere salvati nella folder /work/cv23_group11/dataset
(in quel path va vrata la cartella dataset)

- Gli script devono stare in /homes/utenteServer

- creare la cartella **/dataset** in **/work/proprioGruppo**

- creare job per eseguire gli script python con il comando:
    - **srun -c 12  --mem 10G --time 24:00:00 --partition=students-prod --pty bash**
    - così si ottiene una partizione slurm e da termminale si possono eseguire gli script come di consueto: **python script.py**
    
# Step for dataset generation
- 
- runnare lo script **jpgtopng.py** su 'dataset/train_masks' così avremo tutte le immagini in formato **.png** 
- generare il dataset con lo script in data_generation -> **Random_optimized.py**

- calcolarci media e dev. std con lo script: **mean_std.py**
    - se non va questo script è perhè non si hanno tutte le immagini in formato **.png** 
   

- Il dataset è pronto per essere usato.




