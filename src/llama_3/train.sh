
task='sft'
num_gpus='4'
model_name='Meta-Llama-3-8B-Instruct'

# # Full training with ZeRO-3 on 8 GPUs
ACCELERATE_LOG_LEVEL=info accelerate launch --num_processes 4 --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_$task.py recipes/$model_name/$task/config_full_fold0.yaml

python testllama.py --fold fold_0

ACCELERATE_LOG_LEVEL=info accelerate launch --num_processes 4 --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_$task.py recipes/$model_name/$task/config_full_fold1.yaml

python testllama.py --fold fold_1

ACCELERATE_LOG_LEVEL=info accelerate launch --num_processes 4 --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_$task.py recipes/$model_name/$task/config_full_fold2.yaml

python testllama.py --fold fold_2

ACCELERATE_LOG_LEVEL=info accelerate launch --num_processes 4 --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_$task.py recipes/$model_name/$task/config_full_fold3.yaml

python testllama.py --fold fold_3

ACCELERATE_LOG_LEVEL=info accelerate launch --num_processes 4 --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_$task.py recipes/$model_name/$task/config_full_fold4.yaml

python testllama.py --fold fold_4

# QLoRA 4-bit training on a single GPU
#ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 scripts/run_{task}.py recipes/{model_name}/{task}/config_qlora.yaml


# LoRA training on a single GPU
#ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 scripts/run_{task}.py recipes/{model_name}/{task}/config_qlora.yaml --load_in_4bit=false

# LoRA training with ZeRO-3 on two or more GPUs
#ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml --num_processes={num_gpus} scripts/run_{task}.py recipes/{model_name}/{task}/config_qlora.yaml --load_in_4bit=false

# QLoRA training with FSDP on two or more GPUs
#ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/fsdp+qlora.yaml --num_processes={num_gpus} scripts/run_{task}.py recipes/{model_name}/{task}/config_qlora.yaml --torch_dtype=bfloat16 --bnb_4bit_quant_storage=bfloat16