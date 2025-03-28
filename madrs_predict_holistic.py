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
    0: "madrs_totalscore",
    
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
def requests(data: List[Dict[str, Any]], output_length, prompt_length, model_id, n, tp_size=1) -> List[Dict[str, Any]]:
    """
    Method for deployment batched requests, called once for each batch request.
    """
    if "gemma" in model_id:
        llm = LLM(model=model_id, tensor_parallel_size=tp_size, max_model_len=prompt_length+100, gpu_memory_utilization=1.0)
    else:
        llm = LLM(model=model_id, tensor_parallel_size=tp_size, max_model_len=prompt_length+100)
    sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=500, seed=n)

    full_prompts = [element["messages"] for element in data]
    output_files = [element["output_file"] for element in data]
    labels = [element["label"] for element in data]
    batch_size = 50
    if len(full_prompts) < batch_size:
        batch_size = len(full_prompts)
    for i in tqdm(range(0, len(full_prompts), batch_size), total=len(full_prompts)//batch_size):
        batch_prompts = full_prompts[i:i+batch_size]
        batch_output_files = output_files[i:i+batch_size]
        outputs = llm.generate(batch_prompts, sampling_params)

        for output, output_file, label in zip(outputs, batch_output_files, labels):
            generated_text = output.outputs[0].text  # Assuming only one output is generated per prompt
            with open(output_file, 'w') as file:
                file.write(generated_text)
                file.write(f"\nGround Truth: {label}")
                


def diarize_texts(tokenizer, vtt_transcriptions, output_files, labels, model_id, madrs_idx, n, tp_size=1, prompt_type=""):
    data = []
    max_length = 0
    max_prompt_length = 0
    if prompt_type == "":
        prompt_dir = "full"
    else:
        prompt_dir = f"{prompt_type}"
    prompt_file = f"./prompts/{prompt_dir}/madrs_item_{madrs_idx}.txt"
    with open(prompt_file, 'r') as file:
        prompt = file.read()
    no_test_sessions_file = f"./prompts/{prompt_dir}/madrs_item_{madrs_idx}_no_test_sessions.txt"
    with open(no_test_sessions_file, 'r') as file:
        no_test_sessions = file.read().split("\n")
        no_test_sessions = [session.split("/")[-1] for session in no_test_sessions]
    
    for vtt_transcription, output_file, label in zip(vtt_transcriptions, output_files, labels):
        if output_file.split("/")[-2] in no_test_sessions:
            print(f"Skipping {output_file}")
            continue
        input_text = f"""{prompt}

Your Task:
Transcript:{vtt_transcription}
Output:
Rating:"""
        if model_id not in ["google/gemma-2-27b-it", "mistral-community/Mixtral-8x22B-Instruct-v0.1-4bit"]:
            messages = [
            {"role": "system", "content": "You are a MADRS rater. You will be rating the patient's level of the given MADRS item."},
            {"role": "user", "content": input_text}
            ]
        else:
            messages = [
                {"role": "user", "content": input_text}]
            
        prompts = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        prompt_length = len(tokenizer(prompts)["input_ids"])
        if prompt_length > max_prompt_length:
            max_prompt_length = prompt_length
        input_token_lengths = len(tokenizer(vtt_transcription)["input_ids"])
        # count number of --> in the vtt_transcription
        num_timestamps = vtt_transcription.count("-->")
        
        max_label_token_lengths = max(len(tokenizer("Practitioner: ")["input_ids"]), len(tokenizer("Patient: ")["input_ids"]), len(tokenizer("Unclear: ")["input_ids"]))
        labels_token_lengths = max_label_token_lengths * num_timestamps
        
        output_length = input_token_lengths + labels_token_lengths
        
        if output_length > max_length:
            max_length = output_length
        instance = {"messages": prompts, "output_file": output_file, "label": label}
        data.append(instance)
    
    
    output_length = max_length
    prompt_length = max_prompt_length
    
    requests(data, output_length, prompt_length, model_id, n, tp_size=tp_size)

def process_directory(directory, tokenizer, chunk_no, chunk_idx, model_id, suffix, madrs_idx=2, run_idx=0, tp_size=1, prompt_type=""):
    directories = os.listdir(directory)
    directory_df  = pd.read_csv(f"{directory}/full_dataset_processed.csv")
    
    directories = [patient_dir for patient_dir in directories if os.path.isdir(os.path.join(directory, patient_dir))]
    directories = sorted(directories)    
    vtt_transcriptions = []
    output_files = []
    labels = []
    n = run_idx
    madrs_label = madrs_dict[madrs_idx]
    for dir in tqdm(directories):
        dir_path = os.path.join(directory, dir)
        if not os.path.isdir(dir_path):
            continue
        for session in tqdm([session for session in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, session))]):
            session_path = os.path.join(dir_path, session)
            try:
                label = directory_df[directory_df["video_id"] == session][madrs_label].values[0]
            except:
                continue
            #check if the label is numeric
            try:
                label = int(label)
            except:
                continue
            
            cleaned_transcripts = [os.path.join(session_path, file) for file in os.listdir(session_path) if file.endswith(f"llmQw2.5_72b4q_diarized_cleaned.vtt")]
            
            if len(cleaned_transcripts) == 0:
                continue
            cleaned_transcript = cleaned_transcripts[0]
            vtt_transcription = process_session(cleaned_transcript, tokenizer, model_id, suffix, madrs_idx)
            if vtt_transcription is None or vtt_transcription.strip() == "":
                continue
            if prompt_type != "":
                output_dir = os.path.join(session_path, f"holistic_{suffix}_{prompt_type}")
            else:
                output_dir = os.path.join(session_path, f"holistic_{suffix}")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_file = os.path.join(output_dir, f"madrs{madrs_idx}_output_{n}.txt")
            cont_flag = False
            if os.path.exists(output_file):
                with open(output_file, 'r') as file:
                    output_text = file.read()
                rating = output_text.split("rating:")[-1].split("\n")[0].strip()
                           #get only the number from the string
                rating = ''.join(filter(str.isdigit, rating))
                try:
                    rating = int(rating)
                    cont_flag = True
                except:
                    cont_flag = False
            if cont_flag:
                continue
            vtt_transcriptions.append(vtt_transcription)
            output_files.append(output_file)
            labels.append(label)
    
    chunk_size = len(vtt_transcriptions) // chunk_no
    start_idx = chunk_idx * chunk_size
    end_idx = start_idx + chunk_size if chunk_idx < chunk_no - 1 else len(vtt_transcriptions)
    
    vtt_transcriptions = vtt_transcriptions[start_idx:end_idx]
    output_files = output_files[start_idx:end_idx]
    labels = labels[start_idx:end_idx]
    
    if len(vtt_transcriptions) == 0:
        return
    diarize_texts(tokenizer, vtt_transcriptions, output_files, labels, model_id, madrs_idx, n, tp_size=tp_size, prompt_type=prompt_type)

def process_session(raw_vtt_file, tokenizer, model_id, suffix, madrs_idx):
    with open(raw_vtt_file, 'r') as file:
        vtt_transcription = file.read()
    
    return vtt_transcription


def main():
    parser = argparse.ArgumentParser(description="Process directories for diarization")
    parser.add_argument("--chunk_no", type=int, help="Total number of chunks", default=1)
    parser.add_argument("--chunk_idx", type=int, help="Index of the current chunk", default=0)
    parser.add_argument("--madrs_idx", type=int, help="MADRS item number", default=0)
    parser.add_argument("--model_id", type=str, default= "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4", help="Model ID to use for diarization")
    parser.add_argument("--run_idx", type=int, default=0, help="Run number. Used for multi-run evaluation. Default is 0. Should be less than 5.")
    parser.add_argument("--tp_size", type=int, default=1, help="Number of GPUs for tensor parallelism. Default is 1. If using multiple GPUs, set this to the number of GPUs.")
    parser.add_argument("--prompt_type", type=str, default="", help="Type of prompt to use. Default is empty string for full prompt. Can be used to specify different prompt types for different runs.") 
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
                  "klyang/MentaLLaMA-33B-lora": "MentaLLaMA_33B_lora"
                }
    
    
    if args.run_idx>5:
        print("Run number should be less than 5")
        return
    
    print(f"Processing model {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    from transformers.utils import logging
    logging.set_verbosity_error()

    process_directory(args.directory, tokenizer, args.chunk_no, args.chunk_idx, model_id, model_dict[model_id], madrs_idx=args.madrs_idx, run_idx=args.run_idx, tp_size=args.tp_size, prompt_type=args.prompt_type)

if __name__ == "__main__":
    main()