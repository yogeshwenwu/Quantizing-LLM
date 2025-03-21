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
# sentences = [
#     # "The weather is bad today it might rain anytime",
#     # "Artificial intelligence is transforming the way ",
#     # "The movie I watched yesterday had an unexpected twist at the end ",
#     # "you recommended a good book to read over the weekend, that was",
#     # "The capital of France is Paris, known for its art, culture ",
#     # "She ordered a latte at the café and worked on her presentation ",
#     # "the key differences between machine learning and deep learning is ",
#     # "The traffic on my way to work this morning was ",
#     # "Python is a versatile programming language often used in ",
#     # "He went to the gym every day, determined to improve"
#     "The weather is bad today it might rain anytime",
#     "Artificial intelligence is transforming the way",
#     "The movie I watched yesterday had an unexpected twist at the end",
#     "You recommended a good book to read over the weekend, that was",
#     "The capital of France is Paris, known for its art, culture",
#     "She ordered a latte at the café and worked on her presentation",
#     "The key differences between machine learning and deep learning is",
#     "The traffic on my way to work this morning was",
#     "Python is a versatile programming language often used in",
#     "He went to the gym every day, determined to improve",
#     "Quantum computing has the potential to revolutionize data encryption",
#     "The latest advancements in robotics have enabled autonomous medical procedures",
#     "NASA's new telescope can capture images of distant galaxies with unprecedented clarity",
#     "The discovery of a new exoplanet raises questions about extraterrestrial life",
#     "Meditation has been shown to improve mental clarity and reduce stress",
#     "Scientists recently found a way to generate renewable energy from ocean waves",
#     "The stock market saw a major shift after the latest tech industry boom",
#     "Investing in cryptocurrency can be both rewarding and risky",
#     "Cultural diversity enriches society by introducing new perspectives and traditions",
#     "The human brain remains one of the most complex and least understood organs",
#     "The ethical implications of genetic cloning continue to be widely debated",
#     "If time travel were possible, would we change the past or the future",
#     "The Renaissance period was a time of great artistic and intellectual growth",
#     "She had a dream about a hidden city beneath the ocean waves",
#     "The detective knew the case wasn’t as simple as it seemed",
#     "He found an ancient map hidden inside his grandfather’s journal",
#     "A simple act of kindness can brighten someone’s entire day",
#     "Financial literacy is an essential skill that should be taught in schools",
#     "The invention of the printing press revolutionized the spread of knowledge",
#     "The old man stared at the letter, unsure if he should open it",
#     "The moon shone brighter than ever before, casting an eerie glow over the city",
#     "A mysterious book appeared on her doorstep with no sender address",
#     "Regular exercise not only improves physical health but also boosts mood",
#     "A secret tunnel beneath the library led to a forgotten underground world",
#     "The radio suddenly started playing a song from the future",
#     "The cat stared at the empty space, as if it could see something invisible",
#     "A group of scientists accidentally opened a portal to another dimension",
#     "The future of self-driving cars depends on the reliability of AI decision-making",
#     "The traffic on my way to work this morning was unbearable",
#     "The Eiffel Tower is one of the most famous landmarks in the world",
#     "The clock struck midnight, and the entire town vanished",
#     "He struggled to find the right words to express his gratitude",
#     "Natural language processing allows chatbots to understand human emotions better",
#     "The aroma of freshly brewed coffee filled the cozy café",
#     "Sleep deprivation can negatively impact cognitive function and productivity",
#     "The Great Wall of China was originally built to protect against invasions",
#     "Web development and data science are two of the most popular tech fields today",
#     "Artificial intelligence is expected to revolutionize many industries in the coming years",
#     "The enchanted forest was said to grant wishes to those who entered with pure intentions",
#     "The human body has an incredible ability to heal itself under the right conditions"
# ]
sentences = "In a world where artificial intelligence is transforming every aspect of human life, from automating tasks to revolutionizing scientific research and creative expression, a young developer, driven by an insatiable curiosity and an unwavering determination to push the boundaries of what is possible, embarks on an ambitious journey to create an advanced autograder system, one that not only evaluates code for correctness but also understands the logic behind each implementation, identifies inefficiencies, and provides dynamic, personalized feedback tailored to the unique learning style of each student, ensuring that no two learners receive the same generic responses but instead benefit from an AI-driven mentor capable of analyzing patterns in their mistakes, suggesting alternative approaches, and fostering a deeper understanding of fundamental programming concepts, all while integrating sophisticated natural language processing algorithms to assess short-answer responses with contextual accuracy, allowing students to receive instant, meaningful feedback on theoretical questions alongside their coding assignments, thereby bridging the gap between automated assessment and human-like instruction, creating an intelligent tutoring system that adapts over time, refining its evaluation strategies through continuous learning from a vast dataset of student submissions, detecting trends in conceptual misunderstandings, and dynamically adjusting its grading criteria to align with real-world expectations, ensuring that students are not merely memorizing syntax and standard algorithms but truly grasping the essence of computational thinking, problem decomposition, and optimization, thereby cultivating a new generation of programmers who can think critically, solve complex challenges with confidence, and innovate in ways that traditional education systems often fail to encourage, as the AI-driven platform extends beyond mere evaluation to provide interactive debugging assistance, guiding students through their mistakes with step-by-step explanations, interactive hints, and adaptive learning paths that modify the difficulty of problems based on individual performance, all while seamlessly integrating with existing educational frameworks, allowing instructors to monitor progress, identify struggling students early, and tailor their teaching approaches to address widespread knowledge gaps, making AI not just an evaluator but a true partner in education, one that scales personalized mentorship to levels previously thought impossible, ensuring that every learner, regardless of background or experiences, has access to high-quality guidance that fosters a deep, enduring passion for coding and computational problem-solving"

batch_size = 1#len(sentences)

# >>> Prepare the Model for Quantization <<<
# Set quantization configuration
# model.qconfig = get_default_qconfig(backend)
# # Manually override qconfig for embedding layers to avoid quantization
# for name, module in model.named_modules():
#     if isinstance(module, torch.nn.Embedding):
#         module.qconfig = float_qparams_weight_only_qconfig

quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
# dtype=torch.qint8
# torch.quint8
# torch.qint32
# torch.float16
# torch.bfloat16


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
print("\nRunning benchmark on Native model...")
run_benchmark(model, sentences, tokenizer, device)

print("\nRunning benchmark on Quantized model...")
run_benchmark(quantized_model, sentences, tokenizer, device)