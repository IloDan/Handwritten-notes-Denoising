# Dataset Generation
- the folders with the input images and the grid for the dataset generation are hosted on the AImagelab Server at:
    - '/work/cvcs_2023_group11/input_folder'
    - '/work/cvcs_2023_group11/grids/grid_folder'
      
- Random_optimized.py is the data generation script, which randomly overlays 40 of the total grids on the input images and applies spatial and luminance distortion.
  

# Step for dataset generation
- run the **jpgtopng.py** script on 'dataset/train_masks' so we will have all the images in **.png** format 
- generate the dataset with the script in data_generation -> **Random_optimized.py**

- calculate us mean and dev. std with the script: **mean_std.py**
    - if this script fails it is because you do not have all the images in **.png** format 

- The dataset is ready to be used.

## Use on AImageLab Server
- If working on unimore server:
- All data should be saved in the /work/cv23_group11/dataset folder.
(the dataset folder should be stored in that path).

- Scripts must be in /homes/userServer

- create the **/dataset** folder in **/work/cv23_group**.

- create jobs to run the python scripts with the command:
    - **srun -c 12 --mem 10G --time 24:00:00 --partition=students-prod --pty bash**
    - so you get a slurm partition and from termminal you can run scripts as usual: **python script.py**



