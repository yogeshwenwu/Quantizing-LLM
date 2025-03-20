import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.ao.quantization import get_default_qconfig, prepare, convert, float_qparams_weight_only_qconfig
import time

# >>> Load the tokenizer and model <<<
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
# model = AutoModelForCausalLM.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

device = "cpu"
# model = model.to(device)
# model.eval()


# single sentence
sentences = "In a world where artificial intelligence is transforming every aspect of human life, from automating tasks to revolutionizing scientific research and creative expression, a young developer, driven by an insatiable curiosity and an unwavering determination to push the boundaries of what is possible, embarks on an ambitious journey to create an advanced autograder system, one that not only evaluates code for correctness but also understands the logic behind each implementation, identifies inefficiencies, and provides dynamic, personalized feedback tailored to the unique learning style of each student, ensuring that no two learners receive the same generic responses but instead benefit from an AI-driven mentor capable of analyzing patterns in their mistakes, suggesting alternative approaches, and fostering a deeper understanding of fundamental programming concepts, all while integrating sophisticated natural language processing algorithms to assess short-answer responses with contextual accuracy, allowing students to receive instant, meaningful feedback on theoretical questions alongside their coding assignments, thereby bridging the gap between automated assessment and human-like instruction, creating an intelligent tutoring system that adapts over time, refining its evaluation strategies through continuous learning from a vast dataset of student submissions, detecting trends in conceptual misunderstandings, and dynamically adjusting its grading criteria to align with real-world expectations, ensuring that students are not merely memorizing syntax and standard algorithms but truly grasping the essence of computational thinking, problem decomposition, and optimization, thereby cultivating a new generation of programmers who can think critically, solve complex challenges with confidence, and innovate in ways that traditional education systems often fail to encourage, as the AI-driven platform extends beyond mere evaluation to provide interactive debugging assistance, guiding students through their mistakes with step-by-step explanations, interactive hints, and adaptive learning paths that modify the difficulty of problems based on individual performance, all while seamlessly integrating with existing educational frameworks, allowing instructors to monitor progress, identify struggling students early, and tailor their teaching approaches to address widespread knowledge gaps, making AI not just an evaluator but a true partner in education, one that scales personalized mentorship to levels previously thought impossible, ensuring that every learner, regardless of background or experiences, has access to high-quality guidance that fosters a deep, enduring passion for coding and computational problem-solving"
# sentences = "In a world where artificial intelligence is transforming every aspect of human life, from automating tasks to revolutionizing scientific research and creative expression, a young developer, driven by an insatiable curiosity and an unwavering determination to push the boundaries of what is possible, embarks on an ambitious journey to create an advanced autograder system, one that not only evaluates code for correctness but also understands the logic behind each implementation, identifies inefficiencies, and provides dynamic, personalized feedback tailored to the unique learning style of each student, ensuring that no two learners receive the same generic responses but instead benefit from an AI-driven mentor capable of analyzing patterns in their mistakes, suggesting alternative approaches, and fostering a deeper understanding of fundamental programming concepts, all while integrating sophisticated natural language processing algorithms to assess short-answer responses with contextual accuracy, allowing students to receive instant, meaningful feedback on theoretical questions alongside their coding assignments, thereby bridging the gap between automated assessment and human-like instruction, creating an intelligent tutoring system that adapts over time, refining its evaluation strategies through continuous learning from a vast dataset of student submissions, detecting trends in conceptual misunderstandings, and dynamically adjusting its grading criteria to align with real-world expectations, ensuring that students are not merely memorizing syntax and standard algorithms but truly grasping the essence of computational thinking, problem decomposition, and optimization, thereby cultivating a new generation of programmers who can think critically, solve complex challenges with confidence, and innovate in ways that traditional education systems often fail to encourage, as the AI-driven platform extends beyond mere evaluation to provide interactive debugging assistance, guiding students through their mistakes with step-by-step explanations, interactive hints, and adaptive learning paths that modify the difficulty of problems based on individual performance, all while seamlessly integrating with existing educational frameworks, allowing instructors to monitor progress, identify struggling students early, and tailor their teaching approaches to address widespread knowledge gaps, making AI not just an evaluator but a true partner in education, one that scales personalized mentorship to levels previously thought impossible, ensuring that every learner, regardless of background or experiences, has access to high-quality guidance that fosters a deep, enduring passion for coding and computational problem-solving"
# #input size =  512

# output = " , ultimately paving the way for a more inclusive, more effective, and more efficient technology-driven education system.As a young developer, I have always been fascinated by the endless possibilities of artificial intelligence and its potential to transform the way we learn and teach. My ambitious journey to create an advanced autograder system is driven by a vision of a more personalized, adaptive, and effective education system, one that leverages the power of AI to provide tailored mentorship and support to each and every learner. My system is designed to evaluate code for correctness, but also to understand the logic behind each implementation, identify inefficiencies, and provide dynamic, personalized feedback tailored to the unique learning style of each student. By analyzing patterns in mistakes and suggesting alternative approaches, my AI-driven mentor fosters a deeper understanding of fundamental programming concepts, rather than simply memorizing syntax and standard algorithms. But my system doesn't stop there. Integrated sophisticated natural language processing algorithms allow it to assess short-answer responses with contextual accuracy, providing instant, meaningful feedback on theoretical questions alongside coding assignments. This seamless integration of AI and education bridges the gap between automated assessment and human-like instruction, creating an intelligent tutoring system that adapts over time. Through continuous learning from a vast dataset of student submissions, my system detects trends in conceptual misunderstandings and dynamically adjusts its grading criteria to align with real-world expectations. By cultivating a new generation of programmers who can think critically, solve complex challenges with confidence, and innovate in ways that traditional education systems often fail to encourage, my AI-driven platform extends beyond mere evaluation to provide interactive debugging assistance, guiding students through their mistakes with step-by-step explanations, interactive hints, and adaptive learning paths that modify the difficulty of problems based on individual performance. But my system is more than just an evaluator – it's a true partner in education. By seamlessly integrating with existing educational frameworks, instructors can monitor progress, identify struggling students early, and tailor their teaching approaches to address widespread knowledge gaps. And as my system scales personalized mentorship to levels previously thought impossible, every learner, regardless of background or experiences, has access to high-quality guidance that fosters a deep, enduring passion for coding and computational problem-solving. Ultimately, my AI-driven autograder system has the potential to revolutionize the way we learn and teach, paving the way for a more inclusive, more effective, and more efficient technology-driven education system. [end of text]"
# #output size =  

# sentences = "In F1, where speed, precision, and strategy define legacies, few names shine as brightly as Lewis Hamilton. A driver of extraordinary skill, determination, and resilience, Hamilton has not only shattered records but also transformed the sport, becoming a symbol of excellence and advocacy. Born on January 7, 1985, in Stevenage, England, his journey from a young karting prodigy to becoming a seven-time Formula 1 World Champion is a testament to relentless ambition and groundbreaking achievements. His story is one of perseverance, as he overcame financial struggles and racial barriers to etch his name in motorsport history. Hamilton’s love for racing began at the tender age of six when his father, Anthony Hamilton, gifted him a remote-controlled car. His talent became evident early on, and by the time he was eight, he was already dominating British karting circuits. His father worked multiple jobs to support his racing dreams, and their sacrifices paid off when Lewis caught the attention of McLaren-Mercedes in 1998, securing a spot in their prestigious young driver program. This marked the beginning of an extraordinary path that would lead him to the pinnacle of Formula 1. His Formula 1 debut in 2007 with McLaren was nothing short of spectacular. Partnered with reigning champion Fernando Alonso, Hamilton displayed remarkable composure, finishing on the podium in his very first race at the Australian Grand Prix. Throughout the season, he delivered stellar performances, securing four wins and twelve podiums, narrowly missing out on the championship by just one point to Kimi Räikkönen. His rookie season remains one of the most impressive in F1 history, and he was immediately regarded as a generational talent destined for greatness. Determined to go one step further, Hamilton returned in 2008 with a clear goal: to win the championship. The season saw intense battles, but nothing was more dramatic than the final race in Brazil, where in the last corner of the last lap, he overtook Timo Glock to finish fifth, securing just enough points to win his first world title. At 23 years old, he became the youngest world champion at the time, as well as the first Black driver to achieve this feat, breaking barriers and inspiring millions around the world. While his time at McLaren brought success, Hamilton made a bold and career-defining move in 2013, leaving for Mercedes. At the time, the decision was met with skepticism, as Mercedes had not yet established itself as a dominant force. However, with the introduction of the hybrid power units in 2014, Mercedes emerged as an unstoppable powerhouse, and Hamilton capitalized on their superiority. He won back-to-back championships in 2014 and 2015, engaging in an intense rivalry with teammate Nico Rosberg. The 2016 season saw Hamilton lose the title to Rosberg by just five points, but his hunger for success only grew stronger. From 2017 to 2020, Hamilton entered an era of complete dominance, winning four consecutive world championships and rewriting F1 history. With each victory, he broke longstanding records, surpassing Michael Schumacher’s 91 career wins and eventually reaching 103 wins—the most by any driver in history. He also holds the record for most pole positions (104+), podiums (197+), and laps led, cementing his status as one of the greatest drivers to ever grace the sport. he equaled Schumacher’s record of seven world championships. The 2021 season produced one of the greatest title fights in F1 history, as Hamilton faced Max Verstappen in an intense season-long battle. Both drivers entered the final race in Abu Dhabi tied on points, setting up a winner-takes-all showdown. Hamilton dominated most of the race, but a controversial late-race safety car decision changed everything. On the final lap, Verstappen was given an opportunity to overtake Hamilton due to a race director’s decision that many deemed unfair. The outcome sparked global debate, with many believing Hamilton was unjustly denied a record-breaking eighth title. Despite the heartbreak, he handled the situation with dignity, further solidifying his reputation as a true champion. Hamilton has cemented himself as one of the greatest of all time." 
# 1024

# batch_size = len(sentences)
# print(batch_size)

inputs = tokenizer(output, return_tensors="pt", padding=True).to(device)
# print(inputs['input_ids'])
input_token_count = inputs["input_ids"].shape[1]# * batch_size
print(f"Total input tokens: {input_token_count}")
