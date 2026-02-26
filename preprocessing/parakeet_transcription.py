#!/usr/bin/env python3
"""
Parakeet TDT 0.6B V2 Transcription Script - Patient-based Processing
This script processes all audio tracks for all sessions of a single patient.
Usage: python parakeet_transcription.py <patient_index>
"""

import os
import sys
import json
import gc
import torch
import librosa
import soundfile as sf
import tempfile
from pathlib import Path
import nemo.collections.asr as nemo_asr
import numpy as np
def create_segments_from_pauses(word_timestamps, 
                               min_pause_duration=2.0,
                               max_segment_duration=30.0,  # Force split at 30s
                               min_segment_duration=2.0):   # Don't create tiny segments
    """
    Create segments based on pauses between words and linguistic boundaries.
    """
    if not word_timestamps:
        return []
    
    segments = []
    current_segment_words = []
    segment_start = word_timestamps[0]['start']
    
    # Linguistic markers that often indicate segment boundaries
    sentence_endings = {'.', '!', '?'}
    clause_markers = {',', ';', ':', '—', '–'}
    
    for i, word in enumerate(word_timestamps):
        current_segment_words.append(word['word'])
        
        # Check for pause after this word
        if i < len(word_timestamps) - 1:
            pause_duration = word_timestamps[i + 1]['start'] - word['end']
            segment_duration = word['end'] - segment_start
            
            # Determine if we should split here
            should_split = False
            split_reason = ""
            
            # 1. Long pause detection
            if pause_duration >= min_pause_duration:
                should_split = True
                split_reason = f"pause_{pause_duration:.2f}s"
            
            # 2. Sentence ending with any pause
            elif any(word['word'].rstrip().endswith(punct) for punct in sentence_endings) and pause_duration > 0.2:
                should_split = True
                split_reason = "sentence_end"
            
            # 3. Clause boundary with moderate pause
            elif any(word['word'].rstrip().endswith(punct) for punct in clause_markers) and pause_duration > 0.3:
                if segment_duration >= min_segment_duration:  # Only split if segment is long enough
                    should_split = True
                    split_reason = "clause_boundary"
            
            # 4. Force split if segment is too long
            elif segment_duration >= max_segment_duration:
                should_split = True
                split_reason = "max_duration"
            
            # 5. Natural speech patterns - question followed by pause
            elif '?' in ' '.join(current_segment_words[-5:]) and pause_duration > 0.3:
                should_split = True
                split_reason = "post_question"
            
            if should_split:
                segment_text = ' '.join(current_segment_words)
                segments.append({
                    'segment': segment_text.strip(),
                    'start': segment_start,
                    'end': word['end'],
                    'start_offset': int(segment_start / 0.01),
                    'end_offset': int(word['end'] / 0.01),
                    'split_reason': split_reason  # For debugging
                })
                current_segment_words = []
                segment_start = word_timestamps[i + 1]['start'] if i + 1 < len(word_timestamps) else word['end']
    
    # Add remaining words as final segment
    if current_segment_words:
        segment_text = ' '.join(current_segment_words)
        segments.append({
            'segment': segment_text.strip(),
            'start': segment_start,
            'end': word_timestamps[-1]['end'],
            'start_offset': int(segment_start / 0.01),
            'end_offset': int(word_timestamps[-1]['end'] / 0.01),
            'split_reason': 'final'
        })
    
    return segments


def analyze_word_pauses(word_timestamps):
    """Analyze pause patterns in the audio to help tune segmentation parameters."""
    if len(word_timestamps) < 2:
        return {}
    
    pauses = []
    for i in range(len(word_timestamps) - 1):
        pause = word_timestamps[i + 1]['start'] - word_timestamps[i]['end']
        if pause > 0:
            pauses.append({
                'duration': pause,
                'after_word': word_timestamps[i]['word'],
                'before_word': word_timestamps[i + 1]['word'],
                'position': i
            })
    
    if not pauses:
        return {}
    
    pause_durations = [p['duration'] for p in pauses]
    
    return {
        'total_pauses': len(pauses),
        'mean_pause': np.mean(pause_durations),
        'median_pause': np.median(pause_durations),
        'std_pause': np.std(pause_durations),
        'pause_percentiles': {
            '25th': np.percentile(pause_durations, 25),
            '50th': np.percentile(pause_durations, 50),
            '75th': np.percentile(pause_durations, 75),
            '90th': np.percentile(pause_durations, 90),
            '95th': np.percentile(pause_durations, 95)
        },
        'long_pauses': [p for p in pauses if p['duration'] > 0.5]
    }


def fix_segments_with_intelligent_splitting(transcript_data):
    """
    Fix segments by using word-level timestamps to create better segmentation.
    """
    if 'timestamps' not in transcript_data:
        return transcript_data
    
    # Check if we have word timestamps
    if 'word' not in transcript_data['timestamps'] or not transcript_data['timestamps']['word']:
        print("  Warning: No word timestamps available for intelligent segmentation")
        return transcript_data
    
    word_timestamps = transcript_data['timestamps']['word']
    
    # Analyze pause patterns (optional - for debugging/tuning)
    pause_analysis = analyze_word_pauses(word_timestamps)
    if pause_analysis:
        print(f"  Pause analysis: median={pause_analysis['median_pause']:.3f}s, "
              f"90th percentile={pause_analysis['pause_percentiles']['90th']:.3f}s")
    
    # Create new segments based on pauses
    new_segments = create_segments_from_pauses(
        word_timestamps,
        min_pause_duration=2.0,  # Adjust based on your audio characteristics
        max_segment_duration=30.0,
        min_segment_duration=2.0
    )
    
    # Check if we fixed the long segment issue
    old_segments = transcript_data['timestamps'].get('segment', [])
    if old_segments:
        old_max_duration = max(s['end'] - s['start'] for s in old_segments)
        new_max_duration = max(s['end'] - s['start'] for s in new_segments)
        print(f"  Segmentation improvement: max duration {old_max_duration:.1f}s → {new_max_duration:.1f}s")
        print(f"  Segment count: {len(old_segments)} → {len(new_segments)}")
    
    # Update the transcript data
    transcript_data['timestamps']['segment'] = new_segments
    
    return transcript_data
def find_patients():
    """Find all patient directories in the CAMI structure."""
    base_dir = '/home/gyk/CAMI'
    patient_dirs = []
    
    # Get all patient directories
    if os.path.exists(base_dir):
        for item in os.listdir(base_dir):
            patient_path = os.path.join(base_dir, item)
            if os.path.isdir(patient_path):
                patient_dirs.append(patient_path)
    
    return sorted(patient_dirs)

def find_patient_audio_files(patient_dir):
    """Find all audio files for a specific patient across all sessions."""
    audio_files = []
    
    # Walk through all sessions for this patient
    for session_item in os.listdir(patient_dir):
        session_path = os.path.join(patient_dir, session_item)
        if not os.path.isdir(session_path):
            continue
            
        audio_tracks_dir = os.path.join(session_path, 'audio_tracks')
        if not os.path.exists(audio_tracks_dir):
            continue
            
        # Get all normalized wav files in this session
        track_files = sorted([f for f in os.listdir(audio_tracks_dir) 
                            if f.endswith('_DeepFilterNet3_normalized.wav')])
        
        for track_file in track_files:
            full_path = os.path.join(audio_tracks_dir, track_file)
            audio_files.append(full_path)
    
    return audio_files

def convert_to_16khz(audio_file_path):
    """Convert audio file to 16kHz and return path to converted file."""
    try:
        # Load audio file
        audio, original_sr = librosa.load(audio_file_path, sr=None)
        
        print(f"  Original sample rate: {original_sr} Hz")
        
        # If already 16kHz, return original file
        if original_sr == 16000:
            print(f"  Already 16kHz - using original file")
            return audio_file_path, audio, 16000
        
        # Convert to 16kHz
        audio_16k = librosa.resample(audio, orig_sr=original_sr, target_sr=16000)
        print(f"  Converted to 16kHz")
        
        # Create temporary file for 16kHz version
        base_name = Path(audio_file_path).stem
        temp_dir = Path(audio_file_path).parent
        temp_file = temp_dir / f'{base_name.replace("48k", "16k")}.wav'
        
        # Save 16kHz version
        sf.write(temp_file, audio_16k, 16000)
        
        return str(temp_file), audio_16k, 16000
        
    except Exception as e:
        print(f"  Warning: Could not convert to 16kHz: {e}")
        print(f"  Using original file")
        return audio_file_path, None, None

def cleanup_temp_file(temp_file_path, original_file_path):
    """Clean up temporary 16kHz file if it was created."""
    if temp_file_path != original_file_path and Path(temp_file_path).exists():
        try:
            os.remove(temp_file_path)
            print(f"  Cleaned up temp file: {Path(temp_file_path).name}")
        except Exception as e:
            print(f"  Warning: Could not remove temp file: {e}")

def get_audio_duration(audio_file_path):
    """Get audio duration in seconds."""
    try:
        duration = librosa.get_duration(path=audio_file_path)
        return duration
    except Exception as e:
        print(f"Warning: Could not get duration for {audio_file_path}: {e}")
        return None

def clear_gpu_cache():
    """Clear GPU cache to free memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def create_audio_chunks(audio_data, sample_rate, max_duration_minutes=15):
    """Split audio data into chunks of equal size, all less than maximum duration."""
    max_duration_seconds = max_duration_minutes * 60
    max_samples = int(max_duration_seconds * sample_rate)
    
    total_samples = len(audio_data)
    chunks = []
    
    if total_samples <= max_samples:
        # No chunking needed
        return [(audio_data, 0.0)]
    
    # Calculate number of chunks needed
    n_chunks = int(np.ceil(total_samples / max_samples))
    
    # Calculate base chunk size and remainder
    base_chunk_size = total_samples // n_chunks
    remainder = total_samples % n_chunks
    
    # Ensure no chunk exceeds max_samples
    if base_chunk_size >= max_samples:
        # If base chunk size would exceed max, increase number of chunks
        n_chunks = int(np.ceil(total_samples / max_samples))
        base_chunk_size = total_samples // n_chunks
        remainder = total_samples % n_chunks
    
    print(f"  Splitting into {n_chunks} chunks of approximately {base_chunk_size/sample_rate/60:.1f} minutes each")
    
    current_pos = 0
    for i in range(n_chunks):
        # First 'remainder' chunks get one extra sample to distribute remainder evenly
        chunk_size = base_chunk_size + (1 if i < remainder else 0)
        
        start_sample = current_pos
        end_sample = current_pos + chunk_size
        
        chunk = audio_data[start_sample:end_sample]
        offset_seconds = start_sample / sample_rate
        
        chunks.append((chunk, offset_seconds))
        print(f"    Chunk {i+1}: {len(chunk)/sample_rate/60:.1f} minutes (offset: {offset_seconds/60:.1f} min)")
        
        current_pos = end_sample
    
    return chunks

def merge_transcription_results(chunk_results):
    """Merge transcription results from multiple chunks."""
    merged_text = []
    merged_word_timestamps = []
    merged_segment_timestamps = []
    merged_char_timestamps = []
    
    for chunk_data in chunk_results:
        text, timestamps, offset = chunk_data
        merged_text.append(text)
        
        # Adjust word timestamps
        if 'word' in timestamps:
            for word_data in timestamps['word']:
                adjusted_word = word_data.copy()
                adjusted_word['start'] += offset
                adjusted_word['end'] += offset
                merged_word_timestamps.append(adjusted_word)
        
        # Adjust segment timestamps
        if 'segment' in timestamps:
            for segment_data in timestamps['segment']:
                adjusted_segment = segment_data.copy()
                adjusted_segment['start'] += offset
                adjusted_segment['end'] += offset
                merged_segment_timestamps.append(adjusted_segment)
        
        # Adjust char timestamps
        if 'char' in timestamps:
            for char_data in timestamps['char']:
                adjusted_char = char_data.copy()
                adjusted_char['start'] += offset
                adjusted_char['end'] += offset
                merged_char_timestamps.append(adjusted_char)
    
    # Combine results
    merged_result = {
        'text': ' '.join(merged_text),
        'timestamps': {}
    }
    
    if merged_word_timestamps:
        merged_result['timestamps']['word'] = merged_word_timestamps
    if merged_segment_timestamps:
        merged_result['timestamps']['segment'] = merged_segment_timestamps
    if merged_char_timestamps:
        merged_result['timestamps']['char'] = merged_char_timestamps
    
    return merged_result

def transcribe_with_parakeet(audio_file_path, asr_model):
    """Transcribe audio file using Parakeet model with memory management and chunking."""
    temp_file_path = None
    temp_chunk_files = []
    
    try:
        print(f"Processing: {audio_file_path}")
        
        base_name = Path(audio_file_path).stem
        output_dir = Path(audio_file_path).parent
        
        # Convert to 16kHz and get audio data
        print(f"  Converting to 16kHz...")
        temp_file_path, audio_16k, sample_rate = convert_to_16khz(audio_file_path)
        
        if audio_16k is None:
            # Fallback: load the audio file
            audio_16k, sample_rate = librosa.load(temp_file_path, sr=16000)
        
        # Clear GPU cache before processing
        clear_gpu_cache()
        
        # Check audio duration
        duration = len(audio_16k) / sample_rate
        print(f"  Audio duration: {duration/60:.1f} minutes")
        
        # Create chunks if needed
        chunks = create_audio_chunks(audio_16k, sample_rate, max_duration_minutes=24)
        
        chunk_results = []
        
        # Process each chunk
        for i, (chunk_audio, offset_seconds) in enumerate(chunks):
            if len(chunks) > 1:
                print(f"  Processing chunk {i+1}/{len(chunks)}...")
            
            # Save chunk to temporary file
            chunk_file = output_dir / f"{base_name}_chunk_{i}.wav"
            sf.write(chunk_file, chunk_audio, sample_rate)
            temp_chunk_files.append(chunk_file)
            
            try:
                # Transcribe chunk
                output = asr_model.transcribe([str(chunk_file)], timestamps=True)
                
                if not output or len(output) == 0:
                    print(f"    No output generated for chunk {i+1}")
                    continue
                
                result = output[0]
                
                # Extract timestamps
                timestamps = {}
                if hasattr(result, 'timestamp') and result.timestamp:
                    if 'word' in result.timestamp:
                        timestamps['word'] = result.timestamp['word']
                    if 'segment' in result.timestamp:
                        timestamps['segment'] = result.timestamp['segment']
                    if 'char' in result.timestamp:
                        timestamps['char'] = result.timestamp['char']
                if 'word' in timestamps and timestamps['word']:
                    # Fix any problematic long segments
                    chunk_result_fixed = fix_segments_with_intelligent_splitting({
                        'text': result.text,
                        'timestamps': timestamps
                    })
                    timestamps = chunk_result_fixed['timestamps']
                chunk_results.append((result.text, timestamps, offset_seconds))
                
                if len(chunks) > 1:
                    print(f"    ✓ Chunk {i+1} transcribed successfully")
                
                # Clear cache after each chunk
                clear_gpu_cache()
                
            except torch.cuda.OutOfMemoryError as oom_error:
                # Cleanup
                print(f"❌ Out of memory error processing chunk {i+1}: {str(oom_error)}")
                clear_gpu_cache()
                for chunk_file in temp_chunk_files:
                    if chunk_file.exists():
                        os.remove(chunk_file)
                cleanup_temp_file(temp_file_path, audio_file_path)
                return False
        
        if not chunk_results:
            print(f"No transcription results generated for {audio_file_path}")
            # Cleanup
            for chunk_file in temp_chunk_files:
                if chunk_file.exists():
                    os.remove(chunk_file)
            cleanup_temp_file(temp_file_path, audio_file_path)
            return False
        
        # Merge results if multiple chunks
        if len(chunk_results) > 1:
            print("  Merging chunk results...")
            final_result = merge_transcription_results(chunk_results)
        else:
            # Single chunk - use as is
            text, timestamps, _ = chunk_results[0]
            final_result = {
                'text': text,
                'timestamps': timestamps
            }
        
        # Add metadata
        final_result['sample_rate_converted'] = '16kHz'
        final_result['audio_duration_minutes'] = duration / 60
        if len(chunks) > 1:
            final_result['chunks_processed'] = len(chunks)

        # Save detailed results with timestamps
        json_file = output_dir / f"{base_name}_parakeet.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, indent=2, ensure_ascii=False)
        
        # Create VTT file for compatibility
        vtt_file = output_dir / f"{base_name}_parakeet.vtt"
        create_vtt_file(final_result, vtt_file)
        
        print(f"✓ Successfully transcribed: {base_name}")
        print(f"  JSON saved to: {json_file}")
        print(f"  VTT saved to: {vtt_file}")
        
        # Clear cache after successful processing
        clear_gpu_cache()
        
        # Clean up temporary files
        for chunk_file in temp_chunk_files:
            if chunk_file.exists():
                os.remove(chunk_file)
        cleanup_temp_file(temp_file_path, audio_file_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing {audio_file_path}: {str(e)}")
        # Clear cache on error
        clear_gpu_cache()
        # Clean up temp files on error
        for chunk_file in temp_chunk_files:
            if chunk_file.exists():
                os.remove(chunk_file)
        if temp_file_path:
            cleanup_temp_file(temp_file_path, audio_file_path)
        return False

def create_vtt_file(transcript_data, vtt_file_path):
    """Create a VTT file from transcript data."""
    try:
        with open(vtt_file_path, 'w', encoding='utf-8') as f:
            f.write("WEBVTT\n\n")
            
            # Use segment timestamps if available
            if 'timestamps' in transcript_data and 'segment' in transcript_data['timestamps']:
                for segment in transcript_data['timestamps']['segment']:
                    start_time = format_time(segment['start'])
                    end_time = format_time(segment['end'])
                    text = segment['segment']
                    f.write(f"{start_time} --> {end_time}\n{text}\n\n")
            else:
                # Fallback: create single segment for entire text
                f.write("00:00:00.000 --> 99:59:59.999\n")
                f.write(f"{transcript_data['text']}\n\n")
                
    except Exception as e:
        print(f"Warning: Could not create VTT file: {str(e)}")

def format_time(seconds):
    """Format time in seconds to VTT format (HH:MM:SS.mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

def main():
    if len(sys.argv) != 2:
        print("Usage: python parakeet_transcription.py <patient_index>")
        patient_index = 0
    
    try:
        patient_index = int(sys.argv[1])
    except:
        patient_index = 11
    
    # Find all patient directories
    print("Finding patient directories...")
    patient_dirs = find_patients()
    print(f"Found {len(patient_dirs)} patients")
    
    if patient_index >= len(patient_dirs):
        print(f"Patient index {patient_index} is out of range (0-{len(patient_dirs)-1})")
        sys.exit(1)
    
    # Get the patient directory for this task
    patient_dir = patient_dirs[patient_index]
    patient_id = os.path.basename(patient_dir)
    print(f"Processing patient: {patient_id}")
    
    # Find all audio files for this patient
    audio_files = find_patient_audio_files(patient_dir)
    print(f"Found {len(audio_files)} audio files for patient {patient_id}")
    
    if len(audio_files) == 0:
        print(f"No audio files found for patient {patient_id}")
        sys.exit(0)
    
    # Load Parakeet model
    print("Loading Parakeet TDT 0.6B V2 model...")
    try:
        # Set memory optimization environment variables
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        asr_model = nemo_asr.models.ASRModel.from_pretrained(
            model_name="nvidia/parakeet-tdt-0.6b-v2"
        )
        
        # Move model to GPU and set to eval mode for memory efficiency
        if torch.cuda.is_available():
            asr_model.eval()
            
        print("✓ Model loaded successfully")
        print(f"✓ CUDA memory optimization enabled")
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        sys.exit(1)
    
    # Process all files for this patient
    print(f"Starting transcription for patient {patient_id}...")
    total_files = len(audio_files)
    successful = 0
    failed = 0
    skipped = 0
    
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n[{i}/{total_files}] Processing: {os.path.basename(audio_file)}")
        
        if os.path.exists(audio_file.replace('.wav', '_parakeet.vtt')):
            print(f"  Skipping {audio_file} - already processed")
            skipped += 1
            continue
            
        success = transcribe_with_parakeet(audio_file, asr_model)
        if success:
            successful += 1
        else:
            failed += 1
        
        print(f"  ↳ Progress: {successful} successful, {failed} failed, {skipped} skipped")
    
    # Final summary
    print(f"\n" + "="*60)
    print(f"PATIENT {patient_id} PROCESSING COMPLETE")
    print(f"="*60)
    print(f"Total files: {total_files}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"="*60)
    
    if failed > 0:
        print("❌ Some files failed to process")
        sys.exit(1)
    else:
        print("✓ All files processed successfully")

if __name__ == "__main__":
    main()