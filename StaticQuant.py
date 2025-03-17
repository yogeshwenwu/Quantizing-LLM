import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.ao.quantization import get_default_qconfig, prepare, convert, float_qparams_weight_only_qconfig
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
torch.backends.quantized.engine = backend

# Original model parameter size
original_size = sum(p.numel() for p in model.parameters())


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
model.qconfig = get_default_qconfig(backend)

# Manually override qconfig for embedding layers to avoid quantization
# float_qparams_weight_only_qconfig = quant.float_qparams_weight_only_qconfig
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Embedding):
        module.qconfig = float_qparams_weight_only_qconfig

print("Preparing model for quantization...")
# quant_model_prepared = prepare(model, inplace=False) #Quantization with copy
prepare(model, inplace=True) # Quantizing the original model itself without quantizing the copy

# >>> Calibration with Sample Data <<<
def calibrate_model(model, sentences, tokenizer, device):
    model.eval()
    with torch.no_grad():
        # for sentence in sentences:
        inputs = tokenizer(sentences, return_tensors="pt", padding=True).to(device)
        # Run the model generation to calibrate observers
        model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=50,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True
        )

print("Calibrating model...")
calibrate_model(model, sentences, tokenizer, device)

# >>> Convert to Quantized Model <<<
print("Converting to quantized model...")
# quantized_model = convert(quant_model_prepared, inplace=False)
convert(model.eval(), inplace=True)
# torch.save(model.state_dict(), "quantized_model.pth")
model.to(device)
model.eval()
print("Model quantized successfully!")

# >>> Print Model Size and Data Types <<<
# original_size = sum(p.numel() for p in model.parameters())
# param_dtypes = {p.dtype for p in model.parameters()}
print(f"Original Model Size: {original_size} parameters")
# print(f"Original Parameter Data Types: {param_dtypes}")

# quantized_size = sum(p.numel() for p in quantized_model.parameters())
# qparam_dtypes = {p.dtype for p in quantized_model.parameters()}
# print(f"Quantized Model Size: {quantized_size} parameters")
# print(f"Quantized Parameter Data Types: {qparam_dtypes}")

quantized_size = sum(p.numel() for p in model.parameters())
qparam_dtypes = {p.dtype for p in model.parameters()}
print(f"UnQuantized Model Size: {quantized_size} parameters")
print(f"Quantized Parameter Data Types: {qparam_dtypes}")
print(model)

# >>> Benchmarking Function <<<
def run_benchmark(model, sentences, tokenizer, device):
    inputs = tokenizer(sentences, return_tensors="pt", padding=True).to(device)
    input_token_count = inputs["input_ids"].shape[1] * batch_size
    print(f"Total input tokens: {input_token_count}")

    # Measure encoding time
    start_time = time.time()
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
# print("\nRunning benchmark on original model...")
run_benchmark(model, sentences, tokenizer, device)

# >>> Run Benchmark on Quantized Model <<<
print("\nRunning benchmark on quantized model...")
# run_benchmark(quantized_model, sentences, tokenizer, device)