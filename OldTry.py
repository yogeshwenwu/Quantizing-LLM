# Old code used to inference Llama
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Llama-2-7b-chat-hf"
quantization_config = None

quantized_model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float32, quantization_config=quantization_config)

tokenizer = AutoTokenizer.from_pretrained(model_id)
input_text = "What are we having for dinner?"
input_ids = tokenizer(input_text, return_tensors="pt")

output = quantized_model.generate(**input_ids, max_new_tokens=10)

print(tokenizer.decode(output[0], skip_special_tokens=True))