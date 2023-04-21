#!/bin/bash
#SBATCH --job-name=HAMR_read    # Job name
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=200gb                     # Job memory request
#SBATCH --time=5:00:00               # Time limit hrs:min:sec
#SBATCH --output=/home/siddhant/bhflare/bart_tools/HAMR_input/bhflare/input/log/analysis_out_%j.log   # Standard output and error log
#SBATCH --mail-user=siddhantsolanki321@gmail.com     # Where to send mail

OMP_NUM_THREADS=$SLURM_NTASKS
export PATH=/home/siddhant/anaconda3/bin:$PATH
source /home/${USER}/.bashrc
source activate zaratan-env
conda activate zaratan-env

#srun ./CurrentSheetGeodesics/build/CurrentSheetGeodesics
python reader.py build_ext --inplace
#python pp.py
#python pp.py
#python plot_old_data.py