# Refer to readme.md file

import time
import torch
from ctransformers import AutoModelForCausalLM
from transformers import AutoTokenizer


device= "cpu"
model_id = "meta-llama/Llama-2-7b-chat-hf"
# model_id = "Tap-M/Luna-AI-Llama2-Uncensored"
model_file = "/mnt/mydisk/yogesh/Quantizing-LLM/Llama2_7b_chat_hf_q8_0.gguf"

model = AutoModelForCausalLM.from_pretrained(model_id, model_file=model_file)

# tokenizer = AutoTokenizer.from_pretrained("Tap-M/Luna-AI-Llama2-Uncensored")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
tokenizer.pad_token = tokenizer.eos_token

sentences = "In a world where artificial intelligence is transforming every aspect of human life, from automating tasks to revolutionizing scientific research and creative expression, a young developer, driven by an insatiable curiosity and an unwavering determination to push the boundaries of what is possible, embarks on an ambitious journey to create an advanced autograder system, one that not only evaluates code for correctness but also understands the logic behind each implementation, identifies inefficiencies, and provides dynamic, personalized feedback tailored to the unique learning style of each student, ensuring that no two learners receive the same generic responses but instead benefit from an AI-driven mentor capable of analyzing patterns in their mistakes, suggesting alternative approaches, and fostering a deeper understanding of fundamental programming concepts, all while integrating sophisticated natural language processing algorithms to assess short-answer responses with contextual accuracy, allowing students to receive instant, meaningful feedback on theoretical questions alongside their coding assignments, thereby bridging the gap between automated assessment and human-like instruction, creating an intelligent tutoring system that adapts over time, refining its evaluation strategies through continuous learning from a vast dataset of student submissions, detecting trends in conceptual misunderstandings, and dynamically adjusting its grading criteria to align with real-world expectations, ensuring that students are not merely memorizing syntax and standard algorithms but truly grasping the essence of computational thinking, problem decomposition, and optimization, thereby cultivating a new generation of programmers who can think critically, solve complex challenges with confidence, and innovate in ways that traditional education systems often fail to encourage, as the AI-driven platform extends beyond mere evaluation to provide interactive debugging assistance, guiding students through their mistakes with step-by-step explanations, interactive hints, and adaptive learning paths that modify the difficulty of problems based on individual performance, all while seamlessly integrating with existing educational frameworks, allowing instructors to monitor progress, identify struggling students early, and tailor their teaching approaches to address widespread knowledge gaps, making AI not just an evaluator but a true partner in education, one that scales personalized mentorship to levels previously thought impossible, ensuring that every learner, regardless of background or experiences, has access to high-quality guidance that fosters a deep, enduring passion for coding and computational problem-solving"
max_length = 512
inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
# outputs = model.generate(
#     inputs["input_ids"],
#     # max_new_tokens=512,
#     repetition_penalty=1.13
# )

output = model(prompt=sentences, max_new_tokens=512, repetition_penalty=1.13)
print(output)
















