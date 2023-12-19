#!/bin/bash
#SBATCH --job-name=probing
# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=8 --cpus-per-task=4 --mem=16000M
# we run on the gpu partition and we allocate 1 titanx gpus
#We expect that our program should not run longer than 4 hours
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=04:00:00
#SBATCH --output=/home/jnf811/qinghua/projects_lists/probing_by_analogy/logs/

echo $SLURMD_NODENAME $CUDA_VISIBLE_DEVICES
nvidia-smi



python code/june_3_wiqueen.py --model_name llama-7b --lan en
python code/june_3_wiqueen.py --model_name llama-7b --altering _ex_random_two --lan en
python code/june_3_wiqueen.py --model_name llama-7b --altering _ex_adjacent --lan en
python code/june_3_wiqueen.py --model_name llama-7b --altering _rotate_two_part --lan en


python code/june_3_wiqueen.py --model_name llama-13b  --lan en
python code/june_3_wiqueen.py --model_name llama-13b --altering _ex_random_two --lan en
python code/june_3_wiqueen.py --model_name llama-13b --altering _ex_adjacent --lan en
python code/june_3_wiqueen.py --model_name llama-13b --altering _rotate_two_part --lan en


python code/june_3_wiqueen.py --model_name llama-30b  --lan en
python code/june_3_wiqueen.py --model_name llama-30b --altering _ex_random_two --lan en
python code/june_3_wiqueen.py --model_name llama-30b --altering _ex_adjacent --lan en
python code/june_3_wiqueen.py --model_name llama-30b --altering _rotate_two_part --lan en













