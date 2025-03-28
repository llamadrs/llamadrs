import torch
import os
import json
from tqdm import tqdm
import argparse
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from typing import List, Dict, Any
import pandas as pd

madrs_dict = {
    2: "madrs2_reportedsadness",
    
    1: "madrs1_apparentsadness",
    
    3:  "madrs3_tension",
    
    4:  "madrs4_sleep",
    
    5:  "madrs5_appetite",
    
    6: "madrs6_concentration",
    
    7:  "madrs7_lassitude",
    
    8:  "madrs8_inabilitytofeel",
    
    9: "madrs9_pessimisticthoughts",
    
    10:  "madrs10_suicidalthoughts",
}

panss_dict = {
    1: "p1_delusions",
    2: "p2_conceptualdisorg",
    3: "p3_hallucinations",
    4: "n1_bluntedaffect",
    5: "n4_socialwithdrawal",
    6: "n6_speechflow",
}
 
ymrs_dict = {
    1: "ymrs1_elevatedmood",
    2: "ymrs2_activity_energy",
    3: "ymrs3_sexualinterest",
    4: "ymrs4_sleep",
    5: "ymrs5_irritability",
    6: "ymrs6_speech",
    7: "ymrs7_language",
    8: "ymrs8_content",
    9: "ymrs9_aggressivebehaviour",
    10: "ymrs10_appearance",
    11: "ymrs11_insight"
}

def requests(data: List[Dict[str, Any]], output_length, prompt_length, model_id) -> List[Dict[str, Any]]:
    """
    Method for deployment batched requests, called once for each batch request.
    """
    
    llm = LLM(model=model_id, tensor_parallel_size=2, max_model_len=prompt_length + 5)
    sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=5)

    full_prompts = [element["messages"] for element in data]
    output_files = [element["output_file"] for element in data]
    batch_size = 200
    if len(full_prompts) < batch_size:
        batch_size = len(full_prompts)
    for i in tqdm(range(0, len(full_prompts), batch_size), total=len(full_prompts)//batch_size):
        batch_prompts = full_prompts[i:i+batch_size]
        batch_output_files = output_files[i:i+batch_size]
        outputs = llm.generate(batch_prompts, sampling_params)

        for output, output_file in zip(outputs, batch_output_files):
            generated_text = output.outputs[0].text  # Assuming only one output is generated per prompt
            if not os.path.exists(os.path.dirname(output_file)):
                os.makedirs(os.path.dirname(output_file))
            with open(output_file, 'w', encoding="utf-8") as file:
                file.write(generated_text)
                


def diarize_texts(tokenizer, questions, output_files, contexts, model_id):
    data = []
    max_length = 0
    max_prompt_length = 0
    prompt_file = f"./prompts/madrs_questions_prompt.txt"
    with open(prompt_file, 'r') as file:
        prompt = file.read()
    
    for question, context, output_file in zip(questions, contexts, output_files):
        input_text = f"""{prompt}
""" + "Context: " + context.replace("\\n", "\n") + "\n" + f"Question: {question}" + "\nMADRS Item:"

        messages = [
        {"role": "system", "content": "You are an AI assistant specializing in mental health assessments. Your task is to analyze questions from therapeutic sessions and classify them according to the Montgomery–Åsberg Depression Rating Scale (MADRS) items. Each question should be categorized into the most relevant MADRS item based on its content and intent."},
        {"role": "user", "content": input_text}
        ]

        prompts = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        prompt_length = len(tokenizer(prompts)["input_ids"])
        if prompt_length > max_prompt_length:
            max_prompt_length = prompt_length
        input_token_lengths = len(tokenizer(question)["input_ids"])
        # count number of --> in the vtt_transcription
        
        output_length = input_token_lengths + 2
        
        if output_length > max_length:
            max_length = output_length
        instance = {"messages": prompts, "output_file": output_file}
        data.append(instance)
    
    
    output_length = max_length
    prompt_length = max_prompt_length
    
    requests(data, output_length, prompt_length, model_id)

def process_directory(questions_csv, tokenizer, chunk_no, chunk_idx, model_id, suffix, directory):
    questions_df = pd.read_csv(questions_csv)
    
    questions = questions_df["question"].values
    sessions = questions_df["session_id"].values
    contexts = questions_df["context"].values
    # rank questions per session
    lines = questions_df["line_idx"].values
    output_files = []
    for question, session, line_idx, context in zip(questions, sessions, lines, contexts):
        patient_id = session.split("_")[0]
        session_path = os.path.join(directory, patient_id, session)
        output_file = os.path.join(session_path, "madrs_questions_qwen", f"{line_idx}.txt")
        #if os.path.exists(output_file):
        #    continue
        output_files.append(output_file)
    
    chunk_size = len(questions) // chunk_no
    start_idx = chunk_idx * chunk_size
    end_idx = start_idx + chunk_size if chunk_idx < chunk_no - 1 else len(questions)
    
    questions = questions[start_idx:end_idx]
    output_files = output_files[start_idx:end_idx]
    
    if len(questions) == 0: 
        return
    
    diarize_texts(tokenizer, questions, output_files, contexts, model_id)


def main():
    parser = argparse.ArgumentParser(description="Process directories for diarization")
    parser.add_argument("--chunk_no", type=int, help="Total number of chunks", default=1)
    parser.add_argument("--chunk_idx", type=int, help="Index of the current chunk", default=0)
    parser.add_argument("--model_id", type=str, default= "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4", help="Model ID to use for diarization")
    parser.add_argument("--directory", type=str, help="Directory containing CAMI data", required=True)
    args = parser.parse_args()

    model_id = args.model_id
    model_dict = {"neuralmagic/Meta-Llama-3.1-70B-Instruct-quantized.w4a16": "70b4q",
                  "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w4a16": "8b4q",
                  "neuralmagic/Meta-Llama-3-70B-Instruct-quantized.w4a16": "3_70b4q",
                  "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4": "Qw2.5_72b4q",
                  "Qwen/Qwen2-72B-Instruct-GPTQ-Int4": "Qw2_72b4q",
                    "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4": "Qw2.5_32b4q",
                    "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4": "Qw2.5_14b4q",
                    "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4": "Qw2.5_7b4q",
                    "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4": "Qw2.5_3b4q",
                  "neuralmagic/Mixtral-8x22B-Instruct-v0.1-FP8": "Mix8x22b4q",
                  "google/gemma-2-27b-it": "Gen2_27b4q",
                  "ModelCloud/Mistral-Large-Instruct-2407-gptq-4bit": "Mistral_2407_4Q",
                }
    
                  

    tokenizer = AutoTokenizer.from_pretrained(model_id)


    from transformers.utils import logging
    logging.set_verbosity_error()

    questions_csv = f"{args.directory}/questions_data.csv"
    process_directory(questions_csv, tokenizer, args.chunk_no, args.chunk_idx, model_id, model_dict[model_id], args.directory)

if __name__ == "__main__":
    main()