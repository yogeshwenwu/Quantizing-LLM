import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.ao.quantization import get_default_qconfig, prepare, convert, float_qparams_weight_only_qconfig
import time

# >>> Load the tokenizer and model <<<
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
# model = AutoModelForCausalLM.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# >>> Move model to CPU explicitly and set to evaluation mode <<<
device = "cpu"
model = model.to(device)
model.eval()
backend = "fbgemm"

# >>> Define a dataset of 10 sentences <<<
sentences = [
    "The weather is bad today it might rain anytime",
    "Artificial intelligence is transforming the way ",
    "The movie I watched yesterday had an unexpected twist at the end ",
    "you recommended a good book to read over the weekend, that was",
    "The capital of France is Paris, known for its art, culture ",
    "She ordered a latte at the café and worked on her presentation ",
    "the key differences between machine learning and deep learning is ",
    "The traffic on my way to work this morning was ",
    "Python is a versatile programming language often used in ",
    "He went to the gym every day, determined to improve"
]
batch_size = len(sentences)

# >>> Prepare the Model for Quantization <<<
# Set quantization configuration
# model.qconfig = get_default_qconfig(backend)
# # Manually override qconfig for embedding layers to avoid quantization
# for name, module in model.named_modules():
#     if isinstance(module, torch.nn.Embedding):
#         module.qconfig = float_qparams_weight_only_qconfig

quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

# original_size = sum(p.numel() for p in model.parameters())
# param_dtypes = {p.dtype for p in model.parameters()}
# print(f"Original Model Size: {original_size} parameters")
# print(f"Original Parameter Data Types: {param_dtypes}")

# quantized_size = sum(p.numel() for p in quantized_model.parameters())
# qparam_dtypes = {p.dtype for p in quantized_model.parameters()}
# print(f"Quantized Model Size: {quantized_size} parameters")
# print(f"Quantized Parameter Data Types: {qparam_dtypes}")


def get_model_info(model, label):
    total_params = sum(p.numel() for p in model.parameters())
    dtypes = {p.dtype for p in model.parameters()}
    print(f"{label} Size: {total_params} parameters")
    print(f"{label} Parameter Data Types: {dtypes}")

# Before quantization
get_model_info(model, "Original Model")

# After quantization
# torch.quantization.prepare(model, inplace=True)
# torch.quantization.convert(model, inplace=True)
get_model_info(quantized_model, "Quantized Model")

# for name, param in model.named_parameters():
#      print(f"Parameter: {name}, Shape: {param.shape}, Data Type: {param.dtype}")

for name, param in quantized_model.named_parameters():
     print(f"Parameter: {name}, Shape: {param.shape}, Data Type: {param.dtype}")

# print(model)
print(quantized_model)

# Check linear layer weights
layer = quantized_model.model.layers[0].self_attn.q_proj
print(layer.weight().dtype)  # Should show torch.qint8
print(layer.weight().element_size())  # Should be 1 byte
