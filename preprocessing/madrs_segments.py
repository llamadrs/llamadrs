import torch
import os
import json
from tqdm import tqdm
import argparse
import sglang as sgl
from transformers import AutoTokenizer
from typing import List, Dict, Any
import pandas as pd
import logging
import asyncio
import nest_asyncio

# Apply nest_asyncio for compatibility with Jupyter/nested loops
nest_asyncio.apply()

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

class SegmentationEngine:
    def __init__(self, **kwargs):
        self.engine = sgl.Engine(**kwargs)
        
    async def segment_single(self, prompt, sampling_params, output_file):
        """Process a single question"""
        try:
            result = await self.engine.async_generate(prompt, sampling_params)
            generated_text = result['text'].strip()
            
            # Write output
            if not os.path.exists(os.path.dirname(output_file)):
                os.makedirs(os.path.dirname(output_file))
            with open(output_file, 'w', encoding="utf-8") as file:
                file.write(generated_text)
            
            return output_file, True
            
        except Exception as e:
            logging.error(f"Error processing {output_file}: {e}")
            return output_file, False

async def process_all_files_concurrent(segmentation_engine, prompts, output_files, sampling_params):
    """Process all files concurrently - let SGLang handle optimal batching"""
    
    logging.info(f"Creating {len(prompts)} concurrent tasks")
    
    # Create all tasks at once
    tasks = []
    for prompt, output_file in zip(prompts, output_files):
        task = asyncio.create_task(
            segmentation_engine.segment_single(prompt, sampling_params, output_file)
        )
        tasks.append(task)
    
    # Process results as they complete with proper progress tracking
    successful = 0
    failed = 0
    
    # Use asyncio.as_completed (not tqdm.as_completed) with regular for loop
    with tqdm(total=len(tasks), desc="Processing files") as pbar:
        for coro in asyncio.as_completed(tasks):
            try:
                output_file, success = await coro
                if success:
                    successful += 1
                    logging.debug(f"Successfully processed: {output_file}")
                else:
                    failed += 1
                    logging.warning(f"Failed to process: {output_file}")
                pbar.update(1)
            except Exception as e:
                failed += 1
                logging.error(f"Task failed with exception: {e}")
                pbar.update(1)
    
    logging.info(f"Processing complete: {successful} successful, {failed} failed")
    return successful, failed

async def diarize_texts_async(engine_kwargs, tokenizer, questions, output_files, contexts, model_id):
    """Process questions using async SGLang processing"""
    
    # Create segmentation engine
    segmentation_engine = SegmentationEngine(**engine_kwargs)
    
    # Load prompt
    prompt_file = f"./prompts/madrs_questions_prompt.txt"
    with open(prompt_file, 'r') as file:
        prompt_template = file.read()
    
    # Prepare all prompts
    all_prompts = []
    all_output_files = []
    
    for question, context, output_file in zip(questions, contexts, output_files):
        input_text = f"""{prompt_template}
""" + "Context: " + context.replace("\\n", "\n") + "\n" + f"Question: {question}" + "\nMADRS Item:"

        messages = [
        {"role": "system", "content": "You are an AI assistant specializing in mental health assessments. Your task is to analyze questions from therapeutic sessions and classify them according to the Montgomery–Åsberg Depression Rating Scale (MADRS) items. Each question should be categorized into the most relevant MADRS item based on its content and intent."},
        {"role": "user", "content": input_text}
        ]

        formatted_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, 
                                                       tokenize=False, enable_thinking=False)
        all_prompts.append(formatted_prompt)
        all_output_files.append(output_file)
    
    # Sampling parameters
    sampling_params = {
        "temperature": 0.6,
        "top_p": 0.9,
        "max_new_tokens": 5,
        "top_k": 20,
        "repetition_penalty": 1.0,
    }
    
    # Process everything concurrently
    successful, failed = await process_all_files_concurrent(
        segmentation_engine, all_prompts, all_output_files, sampling_params
    )
    
    return successful, failed

async def process_directory_async(questions_csv, tokenizer, chunk_no, chunk_idx, model_id, suffix, directory, trans_type="whisper"):
    questions_df = pd.read_csv(questions_csv)
    
    questions = questions_df["question"].values
    sessions = questions_df["session_id"].values
    contexts = questions_df["context"].values
    # rank questions per session
    lines = questions_df["line_idx"].values
    output_files = []
    for question, session, line_idx, context in zip(questions, sessions, lines, contexts):
        patient_id = session.split("_")[0]
        session_path = os.path.join(directory, patient_id, "ra_interviews", session)
        output_file = os.path.join(session_path, f"madrs_questions_{trans_type}", f"{line_idx}.txt")
        #if os.path.exists(output_file):
        #    continue
        output_files.append(output_file)
    
    chunk_size = len(questions) // chunk_no
    start_idx = chunk_idx * chunk_size
    end_idx = start_idx + chunk_size if chunk_idx < chunk_no - 1 else len(questions)
    
    questions = questions[start_idx:end_idx]
    output_files = output_files[start_idx:end_idx]
    contexts = contexts[start_idx:end_idx]
    
    if len(questions) == 0: 
        return
    
    # Engine configuration
    engine_kwargs = {
        'model_path': model_id,
        'context_length': 4096,  # Adjust based on your model
        'log_level': 'info',
        'show_time_cost': True,
        'decode_log_interval': 10,
        'log_requests': True,
        'enable_metrics': True
    }
    
    successful, failed = await diarize_texts_async(engine_kwargs, tokenizer, questions, output_files, contexts, model_id)
    logging.info(f"Segmentation results: {successful} successful, {failed} failed")

async def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(description="Process directories for diarization")
    parser.add_argument("--chunk_no", type=int, help="Total number of chunks", default=1)
    parser.add_argument("--chunk_idx", type=int, help="Index of the current chunk", default=0)
    parser.add_argument("--model_id", type=str, default= "Qwen/Qwen3-32B-AWQ", help="Model ID to use for diarization")
    parser.add_argument("--directory", type=str, help="Directory containing CAMI data", default="/home/gyk/CAMI/")
    parser.add_argument("--directory_pro", type=str, help="Directory containing processed CAMI data", default="/home/gyk/CAMI_processing/")
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
                  "Qwen/Qwen3-32B-AWQ": "Qw3_32b4q",
                }
    
    suffix = model_dict.get(model_id, "unknown_model")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    from transformers.utils import logging as transformers_logging
    transformers_logging.set_verbosity_error()

    logging.info(f"Starting MADRS segmentation with SGLang and model: {model_id}")xCV   
    logging.info(f"Processing chunk {args.chunk_idx + 1} of {argxs.chunk_no}")
    for trans_type in ["parakeet"]:
        questions_csv = f"{args.directory_pro}/questions_data_{trans_type}.csv"
        await process_directory_async(questions_csv, tokenizer, args.chunk_no, args.chunk_idx, model_id, suffix, args.directory, trans_type)

if __name__ == "__main__":
    asyncio.run(main())