from datetime import datetime
# imports for the practice (you can add more if you need)
import numpy as np
import matplotlib.pyplot as plt
import os
# import datetime
from tabulate import tabulate
# pytorch
import torch
# torchtext
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, prepare_model_for_kbit_training, PeftModel


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

lora_weights = os.getenv(saved_model_path)

#determine system and user, can be switched
system = "person1"
user = "person2"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(saved_model_path,
    quantization_config=bnb_config
)

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=saved_model_path, add_bos_token=True, 
                                           use_fast=True)
tokenizer.pad_token = tokenizer.unk_token
ft_model = PeftModel.from_pretrained(base_model, saved_model_path)


device = torch.device("cuda")
ft_model.to(device)
ft_model.eval()

conversation_history = ""

def generate_model_response(user_message):
    # Parameters
    global conversation_history
    temperature = 0.5
    max_new_tokens = 100
    repetition_penalty = 1.025 #1.017
    stop = "</s>"
    
    # New user input
    prompt_in = f"<s><{user}>{user_message}</s>"
    conversation_history += prompt_in
    input_ids = tokenizer(conversation_history, return_tensors="pt").to("cuda")
    
    # Generate the model's response
    with torch.no_grad():
        outputs = ft_model.generate(
            **input_ids,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            stop_strings=stop.split(","),
            tokenizer=tokenizer
        )

    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    split_output = decoded_output.split(f"<{system}>")
    response = split_output[-1]
    conversation_history += f"<s><{system}>{response}</s>"
    return response

# Function to append a user message to the file
def append_user_message_to_file(user_message, file_path='conversation.txt'):
    with open(file_path, 'a') as file:
        # Format user message (right-aligned)
        user_bubble = f"{' ' * (40 - len(user_message))}🟩 {user_message} 🟩\n"
        file.write(user_bubble)

# Function to append a model response to the file
def append_model_response_to_file(model_response, file_path='conversation.txt'):
    with open(file_path, 'a') as file:
        # Format model message (left-aligned)
        model_bubble = f"🟦 {model_response} 🟦\n"
        file.write(model_bubble)

# Function to append the current date and time to the file, centered
def append_date_time_to_file(file_path='conversation.txt'):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    centered_time = now.center(80, '-')
    with open(file_path, 'a') as file:
        file.write(f"\n{centered_time}\n\n")

# Main loop to interact with the user
def chat():
    global conversation_history
    # Print and append the current date and time at the start
    append_date_time_to_file()
    #instructions
    print("Start chatting! Type 'exit' to end the conversation.")
    


    while True:
        # Get user input
        user_message = input("")
        
        if user_message.lower() == 'exit':
            print("Ending the conversation.")
            break
        if user_message.lower() == 'exit':
            print("Ending the conversation.")
            break
        
        # Append the user message to the file
        append_user_message_to_file(user_message)
        
        # Generate a response from the model
        model_response = generate_model_response(user_message)
        
        # Append the model response to the file
        append_model_response_to_file(model_response)



if __name__ == "__main__":
    chat()
