import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.ao.quantization import get_default_qconfig, prepare, convert
from torch.quantization import QuantStub, DeQuantStub, float_qparams_weight_only_qconfig
import time

# Load tokenizer and model
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)

# Move model to CPU explicitly and set to evaluation mode
device = "cpu"
model.to(device)
model.eval()

# Add QuantStub and DeQuantStub to the model
class QuantizedModel(torch.nn.Module): # we don't do this
    def __init__(self, base_model):
        super(QuantizedModel, self).__init__()
        self.quant = QuantStub()
        self.model = base_model
        self.dequant = DeQuantStub()
    
    def forward(self, input_ids, attention_mask=None):
        input_ids = self.quant(input_ids)
        output = self.model.generate(input_ids=input_ids, attention_mask=attention_mask)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

quantized_model = QuantizedModel(model)
quantized_model.qconfig = get_default_qconfig("fbgemm")

for name, module in model.named_modules():
    if isinstance(module, torch.nn.Embedding):
        module.qconfig = float_qparams_weight_only_qconfig

# Prepare for quantization
prepare(quantized_model, inplace=True)

# Sample calibration function
def calibrate_model(model, sentences, tokenizer, device):
    model.eval()
    inputs = tokenizer(sentences, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=50,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True
        )

# Calibrate model
sentences = [
    "Artificial intelligence is transforming the world.",
    "Quantum computing has the potential to revolutionize data encryption.",
    "The latest advancements in robotics have enabled autonomous medical procedures."
]
calibrate_model(quantized_model, sentences, tokenizer, device)

# Convert to quantized model
convert(quantized_model, inplace=True)

# Print model summary
print("Quantization completed!")
print(quantized_model)

def run_benchmark(model, sentences, tokenizer, device):
    model.to(device).eval()
    print(torch.backends.quantized.engine)
    batch_size = len(sentences)

    # Measure encoding time
    start_time = time.time()
    inputs = tokenizer(sentences, return_tensors="pt", padding=True).to(device)
    input_token_count = inputs["input_ids"].shape[1] * batch_size
    print(f"Total input tokens: {input_token_count}")
    enc_time = time.time() - start_time

    # TTFT (Time to First Token)
    ttft_start = time.time()
    with torch.inference_mode():#no_grad():
        outputs_ttft = model.generate(
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
        outputs = model.generate(
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

    # Decode generated sequences
    generated_texts = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
    print("\nGenerated Sequences:")
    for i, text in enumerate(generated_texts):
        print(f"Sentence {i+1}: {text}")

    # Time per output token
    tpot = (total_time - ttft_time) / ((total_output_tokens / batch_size) - 1)

    # Metrics
    latency = total_time  # Entire batch
    throughput_tps = total_output_tokens / total_time
    throughput_rps = batch_size / total_time

    # Display Results
    print("\nBenchmark Results:")
    print(f"Batch Size: {batch_size}")
    print(f"Encoding Time: {enc_time * 1000:.4f} ms")
    print(f"Total Time for Batch: {total_time * 1000:.4f} ms")
    print(f"Latency per Batch: {latency * 1000:.4f} ms")
    print(f"Throughput (TPS): {throughput_tps:.2f} tokens/s")
    print(f"Throughput (RPS): {throughput_rps:.2f} requests/s")
    print(f"Time to First Token (TTFT): {ttft_time:.6f} s")
    print(f"Time per Output Token (TPOT): {tpot:.6f} s")

# >>> Run Benchmark on Original Model <<<
print("\nRunning benchmark on quantized model...")
run_benchmark(model, sentences, tokenizer, device)
