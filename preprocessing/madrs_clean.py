import torch
import os
import json
from tqdm import tqdm
import argparse
from transformers import AutoTokenizer
import sglang as sgl
from typing import List, Dict, Any
import pandas as pd
import logging
import asyncio

import nest_asyncio
import glob

# Apply nest_asyncio for compatibility with Jupyter/nested loops
nest_asyncio.apply()


# Increase GPU utilization
os.environ["SGLANG_ATTENTION_BACKEND"] = "flashinfer"

madrs_dict = {
    1: "madrs_01_apparent_sadness",
    2: "madrs_02_reported_sadness", 
    3: "madrs_03_inner_tension",
    4: "madrs_04_reduced_sleep",
    5: "madrs_05_reduced_appetite",
    6: "madrs_06_concentration_difficulties",
    7: "madrs_07_lassitude",
    8: "madrs_08_inability_to_feel",
    9: "madrs_09_pessimistic_thoughts",
    10: "madrs_10_suicidal_thoughts",
}

# Reported Sadness, Inner Tension, Reduced Sleep, Reduced Appetite, Concentration Difficulties, Lassitude (Fatigue), Inability to Feel (Loss of Interest), Pessimistic Thoughts, Suicidal Thoughts
madrs_name_dict = {
    "madrs_01_apparent_sadness": "Apparent Sadness",
    "madrs_02_reported_sadness": "Reported Sadness",
    "madrs_03_inner_tension": "Inner Tension",
    "madrs_04_reduced_sleep": "Reduced Sleep",
    "madrs_05_reduced_appetite": "Reduced Appetite",
    "madrs_06_concentration_difficulties": "Concentration Difficulties",
    "madrs_07_lassitude": "Lassitude (Fatigue)",
    "madrs_08_inability_to_feel": "Inability to Feel (Loss of Interest)",
    "madrs_09_pessimistic_thoughts": "Pessimistic Thoughts",
    "madrs_10_suicidal_thoughts": "Suicidal Thoughts"
}


class SegmentationEngine:
    def __init__(self, model_path, context_length, **kwargs):
        engine_kwargs = {
            'model_path': model_path,
             'json_model_override_args': '{"rope_scaling": { "rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}}',
            'context_length': context_length,
            'log_level': 'info',
            'show_time_cost': True,
            'decode_log_interval': 10,
            'log_requests': True,
            'enable_metrics': True,
            **kwargs
        }
        self.engine = sgl.Engine(**engine_kwargs)
        
    async def segment_single(self, prompt, sampling_params, output_file):
        """Process a single transcription cleaning task"""
        try:
            result = await self.engine.async_generate(prompt, sampling_params)
            generated_text = result['text'].strip()
            
            # Write output
            if not os.path.exists(os.path.dirname(output_file)):
                os.makedirs(os.path.dirname(output_file))
            with open(output_file, 'w', encoding="utf-8") as file:
                file.write(prompt)
                file.write(generated_text)
            
            return output_file, True
            
        except Exception as e:
            logging.error(f"Error processing {output_file}: {e}")
            return output_file, False

async def process_all_files_concurrent(segmentation_engine, task_data, sampling_params):
    """Process all files concurrently - let SGLang handle optimal batching"""
    
    logging.info(f"Creating {len(task_data)} concurrent tasks")
    
    # Create all tasks at once
    tasks = []
    for prompt, output_file, max_new_tokens in task_data:
        # Update sampling params for this specific task
        task_sampling_params = sampling_params.copy()
        task_sampling_params['max_new_tokens'] = max_new_tokens
        
        task = asyncio.create_task(
            segmentation_engine.segment_single(prompt, task_sampling_params, output_file)
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

def read_vtt_content(vtt_file):
    """Read and clean VTT file content"""
    try:
        with open(vtt_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remove VTT header and metadata
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip VTT header, metadata, and empty lines
            if line.startswith('WEBVTT'):
                continue
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    except Exception as e:
        logging.error(f"Error reading VTT file {vtt_file}: {e}")
        return ""

def load_madrs_prompt(prompt_file):
    """Load MADRS cleaning prompt from file"""
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logging.error(f"Error reading prompt file {prompt_file}: {e}")
        return ""

def count_tokens(text, tokenizer):
    """Count actual tokens using the tokenizer"""
    return len(tokenizer.encode(text, add_special_tokens=False))

def calculate_output_length(input_tokens, model_max_context=1010000):
    """Calculate max_new_tokens based on input size - output should be similar to input length"""
    
    # Buffer for safety
    buffer_tokens = 512
    
    # Calculate available tokens for output
    available_for_output = model_max_context - input_tokens - buffer_tokens
    
    # Output should be similar to double of input length, but respect available space
    max_new_tokens = input_tokens * 3
    
    # Set reasonable minimum and maximum
    min_output_tokens = 256
    max_output_tokens = available_for_output
    max_new_tokens = max(max_new_tokens, min_output_tokens)
    max_new_tokens = min(max_new_tokens, max_output_tokens)
    
    return max_new_tokens

def read_both_transcriptions(group_data):
    """Read both whisper and parakeet transcriptions for a group"""
    whisper_content = ""
    parakeet_content = ""
    
    if 'whisper' in group_data:
        whisper_content = read_vtt_content(group_data['whisper'])
    
    if 'parakeet' in group_data:
        parakeet_content = read_vtt_content(group_data['parakeet'])

    return whisper_content, parakeet_content

def create_structured_output_regex():
    """Create a regex pattern for structured output with three required sections"""
    regex = r"Reasoning:.*?Raw Output:.*?Cleaned Output:.*"
    return regex

async def clean_transcriptions_sglang(model_path, model_max_context, tokenizer, file_groups, madrs_prompts_dict):
    """Process VTT files using SGLang concurrent processing"""
    
    # First pass: calculate max input tokens across all files
    logging.info("Calculating maximum input tokens across all files...")
    max_input_tokens = 0
    all_task_data = []
    
    for group_data in file_groups:
        madrs_item = group_data['madrs_item']
        madrs_item_name = group_data['madrs_item_name']
        output_file = group_data['output_file']
        next_item_name = group_data['next_item_name']
        
        # Read combined transcriptions
        whisper_trans, parakeet_trans = read_both_transcriptions(group_data)
        
        if not whisper_trans or not parakeet_trans:
            logging.warning(f"Empty transcription for MADRS item {madrs_item}")
            continue
        
        # Get prompt for this MADRS item
        madrs_prompt = madrs_prompts_dict.get(madrs_item, "")
        if not madrs_prompt:
            logging.warning(f"No prompt found for MADRS item {madrs_item}")
            continue
        
        # Combine prompt with transcription
        full_prompt = f"""INPUT:
- Two transcription versions: Parakeet and Whisper
- Target MADRS item with questions and description  
- Mixed content including multiple MADRS items

OUTPUT FORMAT:
HH:MM:SS.mmm --> HH:MM:SS.mmm Speaker: Content
(No line numbers, chronological order)

PROCESS:
Include reasoning to ensure correct content selection and cleaning. Output Reasoning, Raw Output, and Cleaned Output. Output nothing else.

Include:

- Direct questions about target MADRS item
- Patient responses to those questions REGARDLESS OF RESPONSE QUALITY OR INFORMATIVENESS
- Follow-up clarifications related to item
- Practitioner redirections back to item
- All utterances in item-specific section

MANDATORY STEP: In your reasoning, explicitly identify the FIRST question asked about the target MADRS item. This question marks the beginning of the relevant segment and must NEVER be excluded.

- Look for the earliest timestamp where the practitioner switches the topic to the target item and identify the latest timestamp where they move AWAY from it.
- State clearly in reasoning: "FIRST QUESTION - TARGET ITEM: [timestamp] [speaker]: [content]" and "FIRST QUESTION - NEXT ITEM: [timestamp] [speaker]: [content]"

Exclude only:

- Questions about other MADRS items
- Responses to those questions about other MADRS items
- The conversation is structured, so each item is mostly self-contained even if segments might be in different parts. Remove any content outside of the segments that pertain to the target item.

CRITICAL: Do NOT exclude practitioner utterances or patient responses based on perceived quality, informativeness, or clinical value. Include ALL utterances relevant to target items, even if they are:

- Brief (e.g., "I don't know", "Maybe", "Not really")
- Unclear or incomplete
- Seemingly uninformative
- Non-committal or vague
- Every patient response to a target item question has clinical significance and must be preserved.

STEP 2 - COMPARE, SELECT, MERGE, AND CLEAN

- Compare Parakeet and Whisper versions for each utterance
- The two sources are not always aligned temporally
- Each source might have many duplicates of same utterance. Remove intra-source and inter-source duplicates
- Select the best quality version based on:
  - Clarity of speech
  - Completeness of sentences  
- Merge content where necessary to create a coherent response
- Clean up any transcription errors, ensuring proper grammar and punctuation, diarization errors
- Split merged utterances where multiple speakers have been incorrectly combined into single segments. Separate based on natural conversation flow and speaker changes.

STEP 3 - RESEQUENCE CHRONOLOGICALLY

- Sort by timestamp (adapt across sources)
- Remove numbering
- Ensure conversational flow

EXAMPLES:

EXAMPLE 1:

Input:

Target Item: Reported Sadness
Next Likely Item: Inner Tension

Parakeet:
21. 00:08:20.100 --> 00:08:22.500 Practitioner: Do you appear sad to others?
22. 00:08:22.700 --> 00:08:24.800 Patient: I think so, my family mentions it.
23. 00:08:45.200 --> 00:08:47.800 Practitioner: Have you been feeling sad this week?
24. 00:08:45.600 --> 00:08:47.800 Practitioner: Have you been feeling sad this week?
25. 00:08:48.000 --> 00:08:49.500 Patient: Yes, definitely.
26. 00:08:50.100 --> 00:08:52.300 Practitioner: Are you feeling tense or restless?
27. 00:08:52.500 --> 00:08:54.800 Patient: Very much so, I can't sit still.
45. 00:15:23.150 --> 00:15:25.890 Practitioner: Have you had any trouble concentrating?
205. 00:22:08.406 --> 00:22:34.166 Patient: And it was just frustrating for me and and he was like okay well 
                                        I'll check back with you in two weeks and it sounded like he didn't 
                                        trust me so that made me sad

Whisper:
14. 00:08:19.800 --> 00:08:22.200 Practitioner: Do you appear sad to others?
15. 00:08:22.400 --> 00:08:24.600 Patient: I think so, my family mentions it.
16. 00:08:44.000 --> 00:08:47.00 Practitioner: Have you been feeling sad this week?
17. 00:08:47.200 --> 00:08:49.500 Patient: Yes, definitely.
18. 00:08:49.700 --> 00:08:52.300 Practitioner: Can you describe what that feels like?
19. 00:08:52.500 --> 00:08:58.900 Patient: It's like this heavy feeling in my chest, and I just want to cry 
                                         sometimes.
20. 00:08:59.100 --> 00:09:01.400 Practitioner: Are you feeling tense or restless?
21. 00:09:01.600 --> 00:09:04.100 Patient: Very much so, I can't sit still.
277. 00:22:12.817 --> 00:22:16.380 Patient: And he was like, OK, well, I'll check back with you in two weeks.

Reasoning:
- FIRST QUESTION - TARGET ITEM: 00:08:45.200 --> 00:08:47.800 Practitioner: Have you been feeling sad this week? (Reported Sadness)
- FIRST QUESTION - NEXT ITEM: 00:08:59.100 --> 00:09:01.400 Practitioner: Are you feeling tense or restless? (Inner Tension)
- Exclude: "Do you appear sad to others" - Apparent Sadness item (before Reported Sadness)
- Include: "feeling sad this week" - direct question about target item
- Deduplicate: "Have you been feeling sad this week?" - same question in both sources, but repeated in Parakeet (if duplicates are mistakenly included in final output, please add "EXCLUDE" suffix; e.g: 00:03:14.800 --> 00:03:17.200 Practitioner: How has your sleep been lately?
00:03:15.100 --> 00:03:17.500 Practitioner: How has your sleep been lately? EXCLUDE
- Exclude: "trouble concentrating" - different MADRS item
- Include: "Yes, definitely" - response to sadness question
- Include: Whisper segments 18-19 - follow-up questions and responses about sadness
- Exclude: "Are you feeling tense or restless" - Inner Tension item (after Reported Sadness)
- Exclude: "he was like okay well I'll check back" - not directly about sadness
- Exclude: "that made me sad" - relevant to reported sadness but outside of reported sadness sections

Raw Output:
00:08:44.000 --> 00:08:47.00 Practitioner: Have you been feeling sad this week?
00:08:45.200 --> 00:08:47.800 Practitioner: Have you been feeling sad this week?
00:08:45.600 --> 00:08:47.800 Practitioner: Have you been feeling sad this week?
00:08:47.200 --> 00:08:49.500 Patient: Yes, definitely.
00:08:48.000 --> 00:08:49.500 Patient: Yes, definitely.
00:08:49.700 --> 00:08:52.300 Practitioner: Can you describe what that feels like?
00:08:52.500 --> 00:08:58.900 Patient: It's like this heavy feeling in my chest, and I just want to cry 
                                         sometimes.

Cleaned Output:
00:08:45.200 --> 00:08:47.800 Practitioner: Have you been feeling sad this week?
00:08:48.000 --> 00:08:49.500 Patient: Yes, definitely.
00:08:49.700 --> 00:08:52.300 Practitioner: Can you describe what that feels like?
00:08:52.500 --> 00:08:58.900 Patient: It's like this heavy feeling in my chest, and I just want to cry 
                                         sometimes.

EXAMPLE 2:

Input:

Target Item: Reduced Sleep
Next Likely Item: Reduced Appetite

Parakeet:
10. 00:03:05.100 --> 00:03:07.500 Practitioner: Are you feeling tense or anxious inside?
11. 00:03:07.700 --> 00:03:10.200 Patient: Yes, there's this constant nervousness.
12. 00:03:15.100 --> 00:03:17.500 Practitioner: How has your sleep been lately?
13. 00:03:17.700 --> 00:03:20.300 Patient: Not great, I wake up a lot.
14. 00:03:20.500 --> 00:03:22.800 Practitioner: How many times per night would you say?
15. 00:03:23.000 --> 00:03:26.400 Patient: Maybe three or four times. It's really frustrating because I can't 
                                         get back to sleep easily.
16. 00:03:27.100 --> 00:03:29.500 Practitioner: How is your appetite these days?
17. 00:03:29.700 --> 00:03:32.200 Patient: I'm barely eating, nothing tastes good.
67. 00:12:45.200 --> 00:12:47.800 Practitioner: Have you been feeling hopeless?
68. 00:12:48.000 --> 00:12:50.500 Patient: Sometimes, yes.

Whisper:  
6. 00:03:04.800 --> 00:03:07.200 Practitioner: Are you feeling tense or anxious inside?
7. 00:03:07.400 --> 00:03:09.900 Patient: Yes, theres this constant nervousness.
8. 00:03:14.800 --> 00:03:17.200 Practitioner: How has your sleep been lately?
9. 00:03:17.400 --> 00:03:20.100 Patient: Not great, I wake up alot.
10. 00:03:20.300 --> 00:03:22.600 Practitioner: How many times per night would you say?
11. 00:03:22.800 --> 00:03:27.200 Patient: Maybe 3 or 4 times. Its really frustating cause I cant get back 
                                          to sleep easy.
12. 00:03:27.400 --> 00:03:30.100 Practitioner: Are you taking anything to help with sleep?
13. 00:03:30.300 --> 00:03:33.800 Patient: No, I don't like taking medications if I can avoid it.
14. 00:03:34.000 --> 00:03:36.500 Practitioner: How is your appetite these days?
15. 00:03:36.700 --> 00:03:39.200 Patient: I'm barely eating, nothing tastes good.

Reasoning:
- FIRST QUESTION - TARGET ITEM: 00:03:15.100 --> 00:03:17.500 Practitioner: How has your sleep been lately? (Reduced Sleep)
- FIRST QUESTION - NEXT ITEM: 00:03:34.000 --> 00:03:36.500 Practitioner: How is your appetite these days? (Reduced Appetite) (if mistakenly included to final output, please add "EXCLUDE" suffix; e.g: 00:03:34.000 --> 00:03:36.500 Practitioner: How is your appetite these days? EXCLUDE)
- Exclude: "Are you feeling tense or anxious inside" - Inner Tension item (before Reduced Sleep)
- Include: All sleep-related questions and responses from both sources
- Exclude: "feeling hopeless" - different MADRS item  
- Include: Whisper segments 12-13 - additional follow-up about sleep not in Parakeet
- Exclude: "How is your appetite these days" - Appetite item (after Reduced Sleep)
- Note: Parakeet has better grammar/spelling, Whisper has additional content

Raw Output:
00:03:14.800 --> 00:03:17.200 Practitioner: How has your sleep been lately?  EXCLUDE
00:03:15.100 --> 00:03:17.500 Practitioner: How has your sleep been lately?
00:03:17.400 --> 00:03:20.100 Patient: Not great, I wake up alot. EXCLUDE
00:03:17.700 --> 00:03:20.300 Patient: Not great, I wake up a lot.
00:03:20.300 --> 00:03:22.600 Practitioner: How many times per night would you say? EXCLUDE
00:03:20.500 --> 00:03:22.800 Practitioner: How many times per night would you say?
00:03:22.800 --> 00:03:27.200 Patient: Maybe 3 or 4 times. Its really frustating cause I cant get back 
                                          to sleep easy. EXCLUDE
00:03:23.000 --> 00:03:26.400 Patient: Maybe three or four times. It's really frustrating because I can't 
                                         get back to sleep easily.
00:03:27.400 --> 00:03:30.100 Practitioner: Are you taking anything to help with sleep?
00:03:30.300 --> 00:03:33.800 Patient: No, I don't like taking medications if I can avoid it.

Cleaned Output:
00:03:15.100 --> 00:03:17.500 Practitioner: How has your sleep been lately?
00:03:17.700 --> 00:03:20.300 Patient: Not great, I wake up a lot.
00:03:20.500 --> 00:03:22.800 Practitioner: How many times per night would you say?
00:03:23.000 --> 00:03:26.400 Patient: Maybe three or four times. It's really frustrating because I can't 
                                         get back to sleep easily.
00:03:27.400 --> 00:03:30.100 Practitioner: Are you taking anything to help with sleep?
00:03:30.300 --> 00:03:33.800 Patient: No, I don't like taking medications if I can avoid it.

---

YOUR TURN (Output Reasoning, Raw Output, and Cleaned Output. Output nothing else.)
CRITICAL: Do NOT exclude practitioner utterances or patient responses based on perceived quality, informativeness, or clinical value. Include ALL utterances relevant to target items. 

Exclude only:

- Questions about other MADRS items
- Responses to those questions about other MADRS items
- The conversation is structured, so each item is mostly self-contained even if segments might be in different parts. Remove any content outside of the segments that pertain to the target item.

{madrs_prompt}

Extract segments from the input for the target MADRS item. Begin with your reasoning, then provide initial output, followed by cleaned output:

Input:

Target item: {madrs_item_name}
Next likely item: {next_item_name}


Parakeet:
{parakeet_trans}

Whisper:
{whisper_trans}

Reasoning:"""

        messages = [ {"role": "system", "content": "You are an AI assistant specializing in cleaning psychiatric interview transcriptions. Your task is to extract and clean only the segments relevant to specific MADRS items while maintaining proper formatting and removing redundant content."},
            {"role": "user", "content": full_prompt}
        ]

        formatted_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, 
                                                       tokenize=False)
        
        # Count actual tokens
        input_tokens = count_tokens(formatted_prompt, tokenizer)
        max_input_tokens = max(max_input_tokens, input_tokens)
        
        # Calculate output tokens (similar to input length)
        max_new_tokens = calculate_output_length(input_tokens, model_max_context)
        
        # Store task data: (prompt, output_file, max_new_tokens)
        all_task_data.append((formatted_prompt, output_file, max_new_tokens))
        
        logging.debug(f"MADRS {madrs_item}: {input_tokens} input tokens, {max_new_tokens} output tokens")
    
    if not all_task_data:
        logging.warning("No valid tasks to process")
        return 0, 0
    
    # Calculate required context length based on max input + max output + buffer
    max_output_tokens = max(task[2] for task in all_task_data)
    required_context = max_input_tokens + max_output_tokens + 512  # 512 token buffer
    actual_model_len = min(required_context, model_max_context)
    
    logging.info(f"Max input tokens: {max_input_tokens}")
    logging.info(f"Max output tokens: {max_output_tokens}")
    logging.info(f"Using model length: {actual_model_len}")
    
    # Create single segmentation engine
    segmentation_engine = SegmentationEngine(
        model_path=model_path, 
        context_length=actual_model_len
    )
    
    
    # Create structured output regex pattern
    structured_regex = create_structured_output_regex()
    
    # Base sampling parameters with structured output
    sampling_params = {
        'temperature': 0.1,
        'top_p': 0.9,
        
        'top_k': 20,
        'repetition_penalty': 1.05,
        'max_new_tokens': 512,  # This will be overridden per task
        'regex': structured_regex
    }
    
    # Run async processing
    return asyncio.run(process_all_files_concurrent(
        segmentation_engine, all_task_data, sampling_params
    ))


async def process_directory_async(tokenizer, chunk_no, chunk_idx, model_id, suffix, directory, prompts_dir="/home/gyk/llamadrs/madrs_segments", model_max_context=32768):
    """Process all VTT files in directory structure"""
    
    # Find all VTT files matching the pattern
    vtt_pattern = f"{directory}/*/ra_interviews/*/madrs_segments/*_merged.vtt"
    all_vtt_files = glob.glob(vtt_pattern)
    
    logging.info(f"Found {len(all_vtt_files)} total VTT files")
    
    # Load all MADRS prompts once
    madrs_prompts_dict = {}
    for item_num in madrs_dict.keys():
        prompt_file = os.path.join(prompts_dir, f"prompt{item_num}.txt")
        madrs_prompt = load_madrs_prompt(prompt_file)
        if madrs_prompt:
            madrs_prompts_dict[item_num] = madrs_prompt
        else:
            logging.warning(f"Could not load prompt for MADRS item {item_num}")
    
    # Group files by session and MADRS item
    file_groups = {}  # {session_madrs_key: {'madrs_item': int, 'whisper': path, 'parakeet': path, 'output_file': path}}
    
    for vtt_file in all_vtt_files:
        filename = os.path.basename(vtt_file)
        
        # Extract session, MADRS item, and transcription type
        madrs_item = None
        trans_type = None
        next_item= None
        next_item_name = None
        
        # Determine transcription type
        if "_whisper_merged.vtt" in filename:
            trans_type = "whisper"
        elif "_parakeet_merged.vtt" in filename:
            trans_type = "parakeet"
        else:
            continue
        
        # Determine MADRS item
        for item_num, item_name in madrs_dict.items():
            if item_name in filename:
                madrs_item = item_num
                madrs_item_name = item_name
                if item_num == 10:
                    next_item = None
                    next_item_name = None
                else:
                    next_item = item_num + 1
                    next_item_name = madrs_dict[next_item]
                break
        
        if madrs_item is None:
            logging.warning(f"Could not identify MADRS item for file: {vtt_file}")
            continue
        
        # Create unique key for this session + MADRS item combination
        session_dir = os.path.dirname(vtt_file)
        session_name = session_dir.split(os.sep)[-2]  # Assuming session name is one level up from the VTT file
        session_madrs_key = f"{session_name}_madrs_{madrs_item:02d}"
        
        if next_item_name is None:
            next_item_name = "YMRS Elevated Mood (Euphoria)"
        else:
            next_item_name = madrs_name_dict[next_item_name]
        
        # Initialize group if needed
        if session_madrs_key not in file_groups:
            file_groups[session_madrs_key] = {
                'madrs_item': madrs_item,
                'madrs_item_name': madrs_name_dict[madrs_item_name],
                'next_item': next_item,
                'next_item_name': next_item_name,
                'session_dir': session_dir,
                'session_name': session_name
            }
        
        # Add this transcription type
        file_groups[session_madrs_key][trans_type] = vtt_file
    
    # Filter groups that have at least one transcription and create output paths
    valid_groups = []
    for session_madrs_key, group_data in file_groups.items():
        has_whisper = 'whisper' in group_data
        has_parakeet = 'parakeet' in group_data
        
        if not (has_whisper or has_parakeet):
            continue
        
        madrs_item = group_data['madrs_item']
        session_dir = group_data['session_dir']
        session_name = group_data['session_name']
        
        # Create output file path
        output_dir = session_dir
        item_name = madrs_dict[madrs_item]
        output_filename = f"{session_name}_{item_name}_cleaned.txt"
        output_file = os.path.join(output_dir, output_filename)
        
        # Skip if already processed
        if os.path.exists(output_file):
            continue
        
        # Create final group data for processing
        processing_group = {
            'madrs_item': madrs_item,
            'madrs_item_name': madrs_name_dict[item_name],
            'next_item': group_data['next_item'],
            'next_item_name': group_data['next_item_name'],
            'output_file': output_file
        }
        
        if has_whisper:
            processing_group['whisper'] = group_data['whisper']
        if has_parakeet:
            processing_group['parakeet'] = group_data['parakeet']
            
        valid_groups.append(processing_group)
    
    if len(valid_groups) == 0:
        logging.info("No files to process")
        return
    
    # Apply chunking to the valid groups
    chunk_size = len(valid_groups) // chunk_no
    start_idx = chunk_idx * chunk_size
    end_idx = start_idx + chunk_size if chunk_idx < chunk_no - 1 else len(valid_groups)
    
    chunked_groups = valid_groups[start_idx:end_idx]
    
    if len(chunked_groups) == 0:
        logging.info("No files in this chunk")
        return
    
    logging.info(f"Processing {len(chunked_groups)} MADRS segments in chunk {chunk_idx + 1}/{chunk_no}")
    
    successful, failed = await clean_transcriptions_sglang(
        model_id, model_max_context, tokenizer, chunked_groups, madrs_prompts_dict
    )
    logging.info(f"Cleaning results: {successful} successful, {failed} failed")

async def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(description="Clean MADRS transcription segments")
    parser.add_argument("--chunk_no", type=int, help="Total number of chunks", default=1)
    parser.add_argument("--chunk_idx", type=int, help="Index of the current chunk", default=0)
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-14B-Instruct-1M", help="Model ID to use for cleaning")
    parser.add_argument("--directory", type=str, help="Directory containing CAMI data", default="/home/gyk/CAMI/")
    parser.add_argument("--prompts_dir", type=str, help="Directory containing MADRS prompts", default="/home/gyk/llamadrs/madrs_segments")
    parser.add_argument("--model_max_context", type=int, help="Model's maximum context length", default=1010000)
    args = parser.parse_args()

    model_id = args.model_id
    model_dict = {
        "neuralmagic/Meta-Llama-3.1-70B-Instruct-quantized.w4a16": "70b4q",
        "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w4a16": "8b4q",
        "neuralmagic/Meta-Llama-3-70B-Instruct-quantized.w4a16": "3_70b4q",
        "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4": "Qw2.5_72b4q",
        "Qwen/Qwen2-72B-Instruct-GPTQ-Int4": "Qw2_72b4q",
        "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4": "Qw2.5_32b4q",
        "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4": "Qw2.5_14b4q",
        "Qwen/Qwen2.5-14B-Instruct-1M": "Qw2.5_14b1m",
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

    logging.info(f"Starting MADRS transcription cleaning with SGLang structured output and model: {model_id}")
    logging.info(f"Model max context length: {args.model_max_context}")
    logging.info(f"Processing chunk {args.chunk_idx + 1} of {args.chunk_no}")
    logging.info("Processing both whisper and parakeet transcriptions together with concurrent structured output")
    
    await process_directory_async(
        tokenizer, args.chunk_no, args.chunk_idx, model_id, suffix, 
        args.directory, args.prompts_dir, args.model_max_context
    )

if __name__ == "__main__":
    asyncio.run(main())