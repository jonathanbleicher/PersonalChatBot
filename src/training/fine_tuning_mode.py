# imports for the practice (you can add more if you need)
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from typing import Tuple
from accelerate import Accelerator, FullyShardedDataParallelPlugin
# pytorch
import torch
import transformers
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
from datasets import Dataset, DatasetDict, load_dataset
# torchtext
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.tuners.lora import LoraLayer

seed = 211
np.random.seed(seed)
torch.manual_seed(seed)



########################################################
################## Paste Lines Here#####################
##
##
base_model_id = "yam-peleg/Hebrew-Mistral-7B"
path_to_data = "path/to/data"
saved_model_path = "path/to/save/model"
##
##
########################################################
########################################################

torch.cuda.empty_cache()
tokenizer = AutoTokenizer.from_pretrained(base_model_id,padding_side="left",
        add_eos_token=True,
        add_bos_token=True,)
model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                                                                                    bnb_4bit_use_double_quant=True,
                                                                                                    bnb_4bit_quant_type='nf4',
                                                                                                    bnb_4bit_compute_dtype=torch.bfloat16))

data_set = load_dataset('json', data_files=path_to_data, 
                             split='train')
eval_dataset = data_set.train_test_split(test_size=0.2)

tokenizer.pad_token = tokenizer.unk_token

#preprossesss the data correctly
def generate_and_tokenize_prompt(example):
    result = tokenizer(
        example['input'], # data is just the text in the input field of the json line
        truncation=True,
        max_length=2000,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

tokenise_train_dataset = eval_dataset['train'].map(generate_and_tokenize_prompt)
tokenise_eval_dataset = eval_dataset['test'].map(generate_and_tokenize_prompt)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# LORA
LORA_R = 32 
LORA_ALPHA = 64 
LORA_DROPOUT = 0.07 
LORA_TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,lm_head"  # lora_target_modules

peft_config = LoraConfig(
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    r=LORA_R,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=LORA_TARGET_MODULES.split(","),
)

model_change = get_peft_model(model, peft_config)


fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
)

accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
model_change = accelerator.prepare_model(model_change)


training_args = transformers.TrainingArguments(
                output_dir='./trials_output',
                evaluation_strategy='epoch',
                warmup_steps=2, 
                learning_rate=2.5e-5,
                per_device_train_batch_size=2,
                per_device_eval_batch_size=2,                                                                                       
                weight_decay=0.001,
                save_total_limit=3,
                num_train_epochs=3,
                gradient_accumulation_steps=4,
                gradient_checkpointing=True,
                optim="paged_adamw_8bit",
                logging_dir="./logs",
                logging_steps=50,
                bf16=True
            )

data_collector = transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
trainer =transformers.Trainer(
    model=model_change,
    args=training_args,
    train_dataset=tokenise_train_dataset,
    eval_dataset=tokenise_eval_dataset,
    data_collator=data_collector
)

total_params = sum(p.numel() for p in model_change.parameters())
print(f"Total parameters: {total_params}")
memory_allocated = torch.cuda.memory_allocated()
print(f"Memory allocated: {memory_allocated / (1024 ** 2):.2f} MB")
model_change.print_trainable_parameters()
model_change.config.use_cache = False 

trainer.train()
trainer.model.save_pretrained(saved_model_path)
tokenizer.save_pretrained(saved_model_path)
