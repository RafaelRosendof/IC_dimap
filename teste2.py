import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Especifica a GPU 0 da NVIDIA

timeStart = time.time()

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-2-13b-chat-hf"
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-13b-chat-hf",
    torch_dtype=torch.bfloat16,
    #torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
    #num_hidden_layers=6,
    hidden_size=280,
)

#model.to("cuda")
print("Load model time: ", -timeStart + time.time())

while(True):
    input_str = input('Enter: Olá me explica o que uma função bijetora')
    input_token_length = input('Enter length: 400')

    if(input_str == 'exit'):
        break

    timeStart = time.time()

    inputs = tokenizer.encode(
        input_str,
        return_tensors="pt"
    )

    outputs = model.generate(
        inputs,
        max_new_tokens=int(input_token_length),
        batch_size=1,
    )

    output_str = tokenizer.decode(outputs[0])

    print(output_str)

    print("Time taken: ", -timeStart + time.time())