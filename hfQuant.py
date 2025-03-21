# Using Huggingface Quantized model 
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
# from ctransformers import AutoModelForCausalLM

# model_id = "TheBloke/Luna-AI-Llama2-Uncensored-GGUF"
model_id = "Tap-M/Luna-AI-Llama2-Uncensored"
filename = "./Luna-AI-Llama2-Uncensored.Q5_K_M.gguf"

# tokenizer = AutoTokenizer.from_pretrained("Tap-M/Luna-AI-Llama2-Uncensored")

# # dequantizes the model
tokenizer = AutoTokenizer.from_pretrained(model_id)#, gguf_file=filename)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_id, filename=filename)#, model_file=filename, model_type="llama")
# model = AutoModelForCausalLM.from_pretrained(model_id, filename)# model_file=filename,
    #  model_type="llama",
    #  max_new_tokens=512,
    #  repetition_penalty=1.13,
    #  do_sample=False)

sentences = "In a world where artificial intelligence is transforming every aspect of human life, from automating tasks to revolutionizing scientific research and creative expression, a young developer, driven by an insatiable curiosity and an unwavering determination to push the boundaries of what is possible, embarks on an ambitious journey to create an advanced autograder system, one that not only evaluates code for correctness but also understands the logic behind each implementation, identifies inefficiencies, and provides dynamic, personalized feedback tailored to the unique learning style of each student, ensuring that no two learners receive the same generic responses but instead benefit from an AI-driven mentor capable of analyzing patterns in their mistakes, suggesting alternative approaches, and fostering a deeper understanding of fundamental programming concepts, all while integrating sophisticated natural language processing algorithms to assess short-answer responses with contextual accuracy, allowing students to receive instant, meaningful feedback on theoretical questions alongside their coding assignments, thereby bridging the gap between automated assessment and human-like instruction, creating an intelligent tutoring system that adapts over time, refining its evaluation strategies through continuous learning from a vast dataset of student submissions, detecting trends in conceptual misunderstandings, and dynamically adjusting its grading criteria to align with real-world expectations, ensuring that students are not merely memorizing syntax and standard algorithms but truly grasping the essence of computational thinking, problem decomposition, and optimization, thereby cultivating a new generation of programmers who can think critically, solve complex challenges with confidence, and innovate in ways that traditional education systems often fail to encourage, as the AI-driven platform extends beyond mere evaluation to provide interactive debugging assistance, guiding students through their mistakes with step-by-step explanations, interactive hints, and adaptive learning paths that modify the difficulty of problems based on individual performance, all while seamlessly integrating with existing educational frameworks, allowing instructors to monitor progress, identify struggling students early, and tailor their teaching approaches to address widespread knowledge gaps, making AI not just an evaluator but a true partner in education, one that scales personalized mentorship to levels previously thought impossible, ensuring that every learner, regardless of background or experiences, has access to high-quality guidance that fosters a deep, enduring passion for coding and computational problem-solving"
outputs = model(sentences)
print(outputs)

# device = "cpu"
# model = model.to(device)
# model.eval()

# sentences = [
#     "The weather is bad today it might rain anytime",
#     "Artificial intelligence is transforming the way",
#     "The movie I watched yesterday had an unexpected twist at the end",
#     "In a world where artificial intelligence is transforming every aspect of human life, from automating tasks to revolutionizing scientific research and creative expression, a young developer, driven by an insatiable curiosity and an unwavering determination to push the boundaries of what is possible, embarks on an ambitious journey to create an advanced autograder system, one that not only evaluates code for correctness but also understands the logic behind each implementation, identifies inefficiencies, and provides dynamic, personalized feedback tailored to the unique learning style of each student, ensuring that no two learners receive the same generic responses but instead benefit from an AI-driven mentor capable of analyzing patterns in their mistakes, suggesting alternative approaches, and fostering a deeper understanding of fundamental programming concepts, all while integrating sophisticated natural language processing algorithms to assess short-answer responses with contextual accuracy, allowing students to receive instant, meaningful feedback on theoretical questions alongside their coding assignments, thereby bridging the gap between automated assessment and human-like instruction, creating an intelligent tutoring system that adapts over time, refining its evaluation strategies through continuous learning from a vast dataset of student submissions, detecting trends in conceptual misunderstandings, and dynamically adjusting its grading criteria to align with real-world expectations, ensuring that students are not merely memorizing syntax and standard algorithms but truly grasping the essence of computational thinking, problem decomposition, and optimization, thereby cultivating a new generation of programmers who can think critically, solve complex challenges with confidence, and innovate in ways that traditional education systems often fail to encourage, as the AI-driven platform extends beyond mere evaluation to provide interactive debugging assistance, guiding students through their mistakes with step-by-step explanations, interactive hints, and adaptive learning paths that modify the difficulty of problems based on individual performance, all while seamlessly integrating with existing educational frameworks, allowing instructors to monitor progress, identify struggling students early, and tailor their teaching approaches to address widespread knowledge gaps, making AI not just an evaluator but a true partner in education, one that scales personalized mentorship to levels previously thought impossible, ensuring that every learner, regardless of background or experiences, has access to high-quality guidance that fosters a deep, enduring passion for coding and computational problem-solving",
#     "You recommended a good book to read over the weekend, that was",
#     "The capital of France is Paris, known for its art, culture",
#     "She ordered a latte at the café and worked on her presentation",
#     "The key differences between machine learning and deep learning is",
#     "Python is a versatile programming language often used in",
#     "He went to the gym every day, determined to improve"
# ]


batch_size = 1# len(sentences)

print("Model:\n")
print(model)

# print("\nModel Parameters:\n")
# for name, param in model.named_parameters():
#      print(f"Parameter: {name}, Shape: {param.shape}, Data Type: {param.dtype}")


def get_model_info(model, label):
    total_params = sum(p.numel() for p in model.parameters())
    dtypes = {p.dtype for p in model.parameters()}
    print(f"{label} Size: {total_params} parameters")
    print(f"{label} Parameter Data Types: {dtypes}")

# # Before quantization
get_model_info(model, "Model")

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
            max_new_tokens=512,
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

# # print("\nRunning benchmark on Quantized model...")
# # run_benchmark(quantized_model, sentences, tokenizer, device)