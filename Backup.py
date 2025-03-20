import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.profiler import profile, record_function, ProfilerActivity, schedule
from torch.quantization import QuantStub, DeQuantStub
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

def run_fn(model):
    sample = [
    "The moon shone brighter than ever before, casting an eerie glow over the city",
    "A mysterious book appeared on her doorstep with no sender address",
    "Regular exercise not only improves physical health but also boosts mood",
    "A secret tunnel beneath the library led to a forgotten underground world"
    ]
    inputs = tokenizer(sample, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            max_new_tokens=50,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True
    )

# >>> Quantize the model <<<
quantized_model = torch.ao.quantization.quantize(
    model,
    run_fn=run_fn,
    run_args=(),  # No additional arguments needed
    inplace=False  # Create a new model instead of modifying in place
)

original_size = sum(p.numel() for p in model.parameters())
param_dtypes = {p.dtype for p in model.parameters()}
print(model)
print(original_size)
print(param_dtypes)
# for name, param in model.named_parameters():
#      print(f"Parameter: {name}, Shape: {param.shape}, Data Type: {param.dtype}")


quantized_size = sum(p.numel() for p in quantized_model.parameters())
qparam_dtypes = {p.dtype for p in quantized_model.parameters()}
print(quantized_model)
print(quantized_size)
print(qparam_dtypes)


# print("Running warmup...")
# for _ in range(3):
#     inputs = tokenizer(sentences, return_tensors="pt", padding=True).to(device)
#     with torch.no_grad():
#         _ = model.generate(**inputs, max_new_tokens=50)

# # >>> Prefill time <<<
# # encode time
# start_time = time.time()
# inputs = tokenizer(sentences, return_tensors="pt", padding=True).to(device)
# input_token_count = inputs["input_ids"].shape[1] * batch_size  # Tokens per prompt * batch size
# print(f"Total input tokens: {input_token_count}")
# enc_time = time.time() - start_time
# print(enc_time)

# # ttft time
# ttft_start = time.time()
# with torch.no_grad():
#    outputs = model.generate(
#         inputs['input_ids'],
#         max_new_tokens=1,
#         do_sample=False,
#         return_dict_in_generate=True,
#         output_scores=True
# )
# ttft_time = time.time() - ttft_start

# # decode time
# with torch.no_grad():
#     # Generate outputs
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=50,
#         return_dict_in_generate=True,
#         output_scores=True
#     )

# generated_texts = tokenizer.decode(outputs, skip_special_tokens=True)

# # Measure total time
# total_time = time.time() - start_time

# output_token_count = outputs.sequences.shape[1] - inputs["input_ids"].shape[1]
# total_output_tokens = output_token_count * batch_size #entire batch
# # total_output_tokens = output_token_count #per batch
# # print(outputs)
# print(f"Total output tokens: {total_output_tokens}")


# # Time per output token
# tpot = (total_time - ttft_time) / ((total_output_tokens / batch_size) - 1)

# # 5. Metrics Calculation
# # latency = total_time / batch_size  # per batch
# latency = total_time  # entire batch
# throughput_tps = total_output_tokens / total_time  
# throughput_rps = batch_size / total_time  

# # 6. Display Results
# print("\nBenchmark Results:")
# print(f"Batch Size: {batch_size}")
# print(f"Total Time for Batch: {total_time * 1000:.4f} ms")
# print(f"Latency per Batch: {latency * 1000:.4f} ms")
# print(f"Throughput (TPS): {throughput_tps:.2f} s")
# print(f"Throughput (RPS): {throughput_rps:.2f} s")
# print(f"Time to First Token (TTFT): {ttft_time:.6f} s")
# print(f"Time per Output Token (TPOT): {tpot:.6f} s")

# ---------------------------
# # Profiling the model over 12 iterations to match schedule
# with profile(
#     activities=[ProfilerActivity.CPU],  # Profile CPU activity
#     schedule=schedule(
#         wait=1,   # Skip 1 iteration
#         warmup=2, # Warm up for 2 iterations
#         active=3, # Record 3 iterations
#         repeat=2  # Repeat the cycle twice (total 12 steps)
#     ),
#     record_shapes=True,   # Record tensor shapes
#     profile_memory=True,  # Track memory usage
#     with_stack=True     # To avoid INTERNAL ASSERT error
# ) as prof:
#     for i in range(12):  # 12 iterations to fully cover the schedule
#         sentence = sentences[i % 10]  # Cycle through 10 sentences
#         inputs = tokenizer(sentence, return_tensors="pt")
#         input_ids = inputs["input_ids"].to("cpu")
        
#         with torch.no_grad():
#             with record_function(f"Inference_Sentence_{(i % 10) + 1}"):
#                 outputs = model(input_ids)
#         prof.step()

# # Print profiling results
# print("Top 10 Operations by CPU Time Total:")
# print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

# print("\nTop 2 Operations by Self CPU Time Total:")
# print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=2))

# # Export Chrome trace for visualization
# prof.export_chrome_trace("llama_10_sentences_trace.json")

# # Optional: Uncomment to verify PyTorch version
# # print(f"PyTorch Version: {torch.__version__}")