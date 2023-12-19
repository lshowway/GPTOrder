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

# python /home/fqt170/qinghua/venv_lists/gpt_order/lib/python3.9/site-packages/transformers/models/llama/convert_llama_weights_to_hf.py --input_dir ~/qinghua/project_lists/LLAMA/   --model_size 13B --output_dir ../foundation_models/llama-13b




python code/may_9_gen_zh.py --model_name llama-7b --lan zh
python code/may_9_gen_zh.py --model_name llama-7b --altering _ex_random_two --lan zh
python code/may_9_gen_zh.py --model_name llama-7b --altering _ex_adjacent --lan zh
python code/may_9_gen_zh.py --model_name llama-7b --altering _rotate_two_part --lan zh


python code/may_9_gen_zh.py --model_name llama-13b --lan zh
sleep 180
python code/may_9_gen_zh.py --model_name llama-13b --altering _ex_random_two --lan zh
sleep 180
python code/may_9_gen_zh.py --model_name llama-13b --altering _ex_adjacent --lan zh
sleep 280
python code/may_9_gen_zh.py --model_name llama-13b --altering _rotate_two_part --lan zh
sleep 180


python code/may_9_gen_zh.py --model_name llama-30b --lan zh
sleep 180
python code/may_9_gen_zh.py --model_name llama-30b --altering _ex_random_two --lan zh
sleep 180
python code/may_9_gen_zh.py --model_name llama-30b --altering _ex_adjacent --lan zh
sleep 280
python code/may_9_gen_zh.py --model_name llama-30b --altering _rotate_two_part --lan zh
sleep 180

python code/may_9_gen_zh.py --model_name llama-65b --lan zh
sleep 180
python code/may_9_gen_zh.py --model_name llama-65b --altering _ex_random_two --lan zh
sleep 180
python code/may_9_gen_zh.py --model_name llama-65b --altering _ex_adjacent --lan zh
sleep 280
python code/may_9_gen_zh.py --model_name llama-65b --altering _rotate_two_part --lan zh
sleep 180











