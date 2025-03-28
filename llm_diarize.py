import transformers
import torch
import os
import json
from tqdm import tqdm
import argparse

def diarize_text(pipeline, vtt_transcription, model_id):
    input_text = f"""Task: Diarize all timed utterances from a VTT (Video Text Track) transcription of a psychiatric session. Each utterance should be assigned to one of two speaker roles: Practitioner (MD/RA) or Patient. If the speaker's role is unclear, output 'Unclear'. The output must retain the exact same format as the input VTT file.

Guidelines:
1. Practitioner (MD/RA) typically:
   - Asks questions or seeks clarification
   - Provides explanations or Practitioner opinions
   - May use clinical terminology
   - Guides the conversation or redirects it when necessary
   - Maintains a more neutral, Practitioner tone

2. Patient typically:
   - Answers questions or describes personal experiences
   - Expresses emotions or symptoms
   - May ask for clarification on medical terms
   - Provides detailed or sometimes tangential responses about their life
   - Shows a range of emotions in their speech

3. Context Interpretation:
   - Consider each utterance within the full context of the transcription
   - Analyze content and style to determine the likely speaker
   - Pay attention to conversational patterns, topic shifts, and emotional tones
   - Consider the overall conversation flow to identify speaker roles

Format:
- The output should match the exact VTT format of the input, including timestamps and text.
- Prepend the corresponding speaker role (Practitioner, Patient, or Unclear) to each utterance while preserving the VTT structure.
- If there is any ambiguity, prepend 'Unclear' to the utterance.

Now, diarize the following VTT transcription and only output the diarized text:
Example:
Input:
...
03:13.098 --> 03:16.600
So a couple things before we get started.
...
Output:
...
03:13.098 --> 03:16.600
Practitioner: So a couple things before we get started.
...

Your turn:
Input:
{vtt_transcription}

Output: """

    if "Llama" in model_id:
        messages = [
            {"role": "system", "content": "You are a diarization model for psychiatric sessions."},
            {"role": "user", "content": input_text}
        ]

        terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
        input_token_lengths = len(pipeline.tokenizer(vtt_transcription)["input_ids"])
        # count number of --> in the vtt_transcription
        num_timestamps = vtt_transcription.count("-->")
        
        max_label_token_lengths = max(len(pipeline.tokenizer("Practitioner: ")["input_ids"]), len(pipeline.tokenizer("Patient: ")["input_ids"]), len(pipeline.tokenizer("Unclear: ")["input_ids"]))
        labels_token_lengths = max_label_token_lengths * num_timestamps
        
        output_length = input_token_lengths + labels_token_lengths
        outputs = pipeline(
            messages,
            max_new_tokens= output_length,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9
        )
    else:
        messages = [
            {"role": "user", "content": input_text}
        ]

        outputs = pipeline(
            messages,
            max_new_tokens=output_length,
            do_sample=True,
            temperature=0.6,
            top_p=0.9)
        
    return outputs[0]["generated_text"][-1]["content"].strip()

def process_directory(directory, pipeline, chunk_no, chunk_idx, model_id):
    directories = os.listdir(directory)
    directories = [patient_dir for patient_dir in directories if os.path.isdir(os.path.join(directory, patient_dir)) 
                and any(os.path.exists(os.path.join(directory, patient_dir, session, file)) 
                        and file.endswith("_raw.vtt") 
                        and not os.path.exists(os.path.join(directory, patient_dir, session, file.replace("_raw.vtt", "_llm_diarized.vtt"))) 
                        for session in os.listdir(os.path.join(directory, patient_dir)) 
                        if os.path.isdir(os.path.join(directory, patient_dir, session)) 
                        for file in os.listdir(os.path.join(directory, patient_dir, session)))]
    directories = sorted(directories)
    chunk_size = len(directories) // chunk_no
    start_idx = chunk_idx * chunk_size
    end_idx = start_idx + chunk_size if chunk_idx < chunk_no - 1 else len(directories)
    print(f"Processing {directory} from {start_idx} to {end_idx}")
    for dir in tqdm(directories[start_idx:end_idx]):
        dir_path = os.path.join(directory, dir)
        if not os.path.isdir(dir_path):
            continue
        for session in tqdm([session for session in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, session))]):
            session_path = os.path.join(dir_path, session)
            raw_vtt_files = [os.path.join(session_path, file) for file in os.listdir(session_path) if file.endswith("_raw.vtt")]
            if len(raw_vtt_files) == 0:
                continue
            raw_vtt_file = raw_vtt_files[0]
            process_session(raw_vtt_file, pipeline, model_id)

def process_session(raw_vtt_file, pipeline, model_id):
    llm_diarized_file = raw_vtt_file.replace("_raw.vtt", "_llm_diarized.vtt")
    if os.path.exists(llm_diarized_file):
        return
    with open(raw_vtt_file, 'r') as file:
        vtt_transcription = file.read()
    vtt_diarized = diarize_text(pipeline, vtt_transcription, model_id)
    with open(llm_diarized_file, 'w') as file:
        file.write(vtt_diarized)


def main():
    parser = argparse.ArgumentParser(description="Process directories for diarization")
    parser.add_argument("--chunk_no", type=int, help="Total number of chunks", default=1)
    parser.add_argument("--chunk_idx", type=int, help="Index of the current chunk", default=0)
    parser.add_argument("--model_id", type=str, default= "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4", help="Model ID to use for diarization")
    parser.add_argument("--directory", type=str, help="Directory containing CAMI data", required=True)
    args = parser.parse_args()

    model_id = args.model_id
    pipeline = transformers.pipeline(
        "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device=0)
    
    from transformers.utils import logging
    logging.set_verbosity_error()

    process_directory(args.directory, pipeline, args.chunk_no, args.chunk_idx, args.model_id.split("/")[-1].replace("-", "_"))

if __name__ == "__main__":
    main()