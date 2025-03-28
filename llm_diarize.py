import transformers
import torch
import os
import json
from tqdm import tqdm
import argparse

def diarize_text(pipeline, context, text, model_id):
    input_text = f"""Task: Given an undiarized verbatim transcription of utterances from a psychiatric session, diarize the specified utterance into one of two speaker roles: Professional (MD/RA) or Patient. If the speaker's role is unclear, output 'Unclear'.

Guidelines:
1. Professional (MD/RA) typically:
   - Asks questions or seeks clarification
   - Provides explanations or professional opinions
   - May use clinical terminology
   - Guides the conversation or redirects it when necessary
   - Maintains a more neutral, professional tone

2. Patient typically:
   - Answers questions or describes personal experiences
   - Expresses emotions or symptoms
   - May ask for clarification on medical terms
   - Provides detailed or sometimes tangential responses about their life
   - Shows a range of emotions in their speech

3. Context Interpretation:
   - The context includes utterances that immediately precede and follow the text to diarize
   - Both the context and the text to diarize are undiarized transcriptions
   - Analyze the content and style of each utterance to determine the likely speaker
   - Consider the overall flow of conversation to identify potential speaker changes
   - Pay attention to question-answer patterns, topic introductions, and shifts in conversation focus
   - In some cases, the transcript only contains one speaker's utterances. In such cases, do not hesitate to assign the same speaker role to all utterances

Output only the role of the speaker (Professional, Patient, or Unclear) without explanation. If there is any ambiguity in the speaker role, output 'Unclear'.

Now, diarize the following utterance based on the given undiarized context:

Context:
{context}

Text: {text}
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

        outputs = pipeline(
            messages,
            max_new_tokens=5,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
    else:
        messages = [
            {"role": "user", "content": input_text}
        ]

        outputs = pipeline(
            messages,
            max_new_tokens=5,
            do_sample=True,
            temperature=0.6,
            top_p=0.9)
        
    return outputs[0]["generated_text"][-1]["content"].strip().replace("\n", " ").lower()

def process_directory(directory, pipeline, chunk_no, chunk_idx, model_id):
    directories = os.listdir(directory)
    directories = [patient_dir for patient_dir in directories if  os.path.isdir(os.path.join(directory, patient_dir))
                   and not all([os.path.exists(os.path.join(directory, patient_dir, session, f"transcription_{model_id}.json")) for session in os.listdir(os.path.join(directory, patient_dir))
                                if os.path.isdir(os.path.join(directory, patient_dir, session))])]
    directories = sorted(directories)
    chunk_size = len(directories) // chunk_no
    start_idx = chunk_idx * chunk_size
    end_idx = start_idx + chunk_size if chunk_idx < chunk_no - 1 else len(directories)
    
    for dir in tqdm(directories[start_idx:end_idx]):
        dir_path = os.path.join(directory, dir)
        if not os.path.isdir(dir_path):
            continue
        for session in os.listdir(dir_path):
            session_path = os.path.join(dir_path, session)
            if not os.path.isdir(session_path) or os.path.exists(os.path.join(session_path, f"transcription_{model_id}.json")):
                continue
            process_session(session_path, pipeline, model_id)

def process_session(session_path, pipeline, model_id):
    transcript_files = [os.path.join(session_path, file) for file in os.listdir(session_path) 
                        if file.startswith("audio_track") and "reduced" in file and file.endswith(".txt")]
    transcript_dict = {}
    
    for file in transcript_files:
        with open(file, "r") as f:
            transcript = f.readlines()
        
        last_line = ""
        for line_no, line in enumerate(transcript):
            line = line.strip()
            time_stamp, text = line.split(": ")
            if text == last_line:
                continue
            start_time, end_time = time_stamp.split("-")
            start_time, end_time = float(start_time), float(end_time)
            
            if f"{start_time}-{end_time}" not in transcript_dict:
                transcript_dict[f"{start_time}-{end_time}"] = []
            
            context = get_context(transcript, line_no)
            speaker_role = diarize_text(pipeline, context, text, model_id)
            
            formatted_text = f"{os.path.basename(file).replace('_reduced_vad.txt', '').replace('_louder', '')}, {speaker_role}: {text}"
            if formatted_text not in transcript_dict[f"{start_time}-{end_time}"]:
                transcript_dict[f"{start_time}-{end_time}"].append(formatted_text)
            
            last_line = text
    
    # save json
    with open(os.path.join(session_path, f"transcription_{model_id}.json"), "w") as f:
        json.dump(transcript_dict, f)

def get_context(transcript, line_no):
    context = []
    context_line_no = max(0, line_no - 5)
    last_context_line = ""
    while len(context) < 11 and context_line_no < len(transcript):
        current_line = transcript[context_line_no].strip()
        if current_line.split(": ")[1] != last_context_line:
            context.append(current_line)
            last_context_line = current_line.split(": ")[1]
        context_line_no += 1
    return "\n".join(context)

def main():
    parser = argparse.ArgumentParser(description="Process directories for diarization")
    parser.add_argument("--chunk_no", type=int, required=True, help="Total number of chunks")
    parser.add_argument("--chunk_idx", type=int, required=True, help="Index of the current chunk")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4", help="Model ID to use for diarization")
    args = parser.parse_args()

    model_id = args.model_id
    pipeline = transformers.pipeline(
        "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device=0)
    
    from transformers.utils import logging
    logging.set_verbosity_error()

    directory = '/home/gyk/CAMI/'
    process_directory(directory, pipeline, args.chunk_no, args.chunk_idx, args.model_id.split("/")[-1].replace("-", "_"))

if __name__ == "__main__":
    main()