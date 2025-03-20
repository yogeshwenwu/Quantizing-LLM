import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.ao.quantization import QuantStub, DeQuantStub, get_default_qconfig, prepare, convert
import time

# >>> Load the tokenizer and model <<<
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
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

# >>> Add Quantization Stubs to the Model <<<
class QuantizedModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.quant = QuantStub()  # Quantize input
        self.model = model
        self.dequant = DeQuantStub()  # Dequantize output

    def forward(self, input_ids, attention_mask=None):
        x = self.quant(input_ids)
        outputs = self.model.generate(x,
            attention_mask=inputs['attention_mask'],
            max_new_tokens=50,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True
            )
        return self.dequant(outputs.sequences)  # Only quantize logits for simplicity

# Wrap the original model
quant_model = QuantizedModel(model)

# >>> Prepare the Model for Quantization <<<
quant_model.qconfig = get_default_qconfig(backend)  # Use 'fbgemm' for x86 CPU
quant_model_prepared = prepare(quant_model, inplace=False)

# >>> Calibration with Sample Data <<<
def calibrate_model(model, sentences, tokenizer, device):
    model.eval()
    with torch.no_grad():
        for sentence in sentences:
            inputs = tokenizer(sentence, return_tensors="pt", padding=True).to(device)
            model(inputs["input_ids"], attention_mask=inputs["attention_mask"])

print("Calibrating model...")
calibrate_model(quant_model_prepared, sentences, tokenizer, device)

# >>> Convert to Quantized Model <<<
quantized_model = convert(quant_model_prepared, inplace=False)
print("Model quantized successfully!")

# >>> Print Model Size and Data Types <<<
original_size = sum(p.numel() for p in model.parameters())
param_dtypes = {p.dtype for p in model.parameters()}
print(f"Original Model Size: {original_size} parameters")
print(f"Original Parameter Data Types: {param_dtypes}")

quantized_size = sum(p.numel() for p in quantized_model.parameters())
qparam_dtypes = {p.dtype for p in quantized_model.parameters()}
print(f"Quantized Model Size: {quantized_size} parameters")
print(f"Quantized Parameter Data Types: {qparam_dtypes}")

# >>> Benchmarking Function <<<
def run_fn(model, sentences, tokenizer, device):
    inputs = tokenizer(sentences, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=50,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True
        )
    return outputs

# >>> Warmup <<<
print("Running warmup...")
for _ in range(3):
    run_fn(quantized_model, sentences[:2], tokenizer, device)  # Warmup with 2 sentences

# >>> Benchmarking <<<
start_time = time.time()
inputs = tokenizer(sentences, return_tensors="pt", padding=True).to(device)
input_token_count = inputs["input_ids"].shape[1] * batch_size
print(f"Total input tokens: {input_token_count}")
enc_time = time.time() - start_time
print(f"Encoding time: {enc_time:.6f} s")

# TTFT (Time to First Token)
ttft_start = time.time()
with torch.no_grad():
    outputs = quantized_model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_new_tokens=1,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True
    )
ttft_time = time.time() - ttft_start

# Full generation
with torch.no_grad():
    outputs = quantized_model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_new_tokens=50,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True
    )

total_time = time.time() - start_time
output_token_count = outputs.sequences.shape[1] - inputs["input_ids"].shape[1]
total_output_tokens = output_token_count * batch_size
print(f"Total output tokens: {total_output_tokens}")

# Time per output token
tpot = (total_time - ttft_time) / ((total_output_tokens / batch_size) - 1)

# Metrics
latency = total_time  # Entire batch
throughput_tps = total_output_tokens / total_time
throughput_rps = batch_size / total_time

# Display Results
print("\nBenchmark Results (Quantized Model):")
print(f"Batch Size: {batch_size}")
print(f"Total Time for Batch: {total_time * 1000:.4f} ms")
print(f"Latency per Batch: {latency * 1000:.4f} ms")
print(f"Throughput (TPS): {throughput_tps:.2f} tokens/s")
print(f"Throughput (RPS): {throughput_rps:.2f} requests/s")
print(f"Time to First Token (TTFT): {ttft_time:.6f} s")
print(f"Time per Output Token (TPOT): {tpot:.6f} s")