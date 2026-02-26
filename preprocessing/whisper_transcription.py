#!/usr/bin/env python3
"""
WhisperX Transcription Script - Patient-based Processing
This script processes all audio tracks for all sessions of a single patient using WhisperX.
Usage: python whisper_transcription.py <patient_index>
"""

import os
import sys
import json
import subprocess
from pathlib import Path

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

def check_existing_whisper_output(audio_file_path):
    """Check if WhisperX transcription already exists."""
    base_name = Path(audio_file_path).stem
    output_dir = Path(audio_file_path).parent
    
    # Check for WhisperX output formats
    output_files = [
        output_dir / f"{base_name}.vtt",
        output_dir / f"{base_name}.json",
    ]
    
    return all(f.exists() for f in output_files)

def transcribe_with_whisperx(audio_file_path):
    """Transcribe audio file using WhisperX."""
    try:
        print(f"Processing with WhisperX: {audio_file_path}")
        
        base_name = Path(audio_file_path).stem
        output_dir = Path(audio_file_path).parent
        
        # Check if already processed
        if check_existing_whisper_output(audio_file_path):
            print(f"WhisperX output already exists for {base_name}. Skipping...")
            return True
        
        # Run WhisperX command
        print("Transcribing with WhisperX...")
        
        cmd = [
            "whisperx",
            str(audio_file_path),
            "--model", "large-v2",
            "--language", "en", 
            "--output_dir", str(output_dir),
            "--compute_type", "float16"
        ]
        
        # Run the command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            print(f"❌ WhisperX failed for {audio_file_path}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
        
        print(f"✓ Successfully transcribed with WhisperX: {base_name}")
        
        # List generated files
        expected_files = [
            output_dir / f"{base_name}.vtt",
            output_dir / f"{base_name}.json"
        ]
        
        for file_path in expected_files:
            if file_path.exists():
                print(f"  Generated: {file_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing with WhisperX {audio_file_path}: {str(e)}")
        return False

def main():
    if len(sys.argv) != 2:
        print("Usage: python whisper_transcription.py <patient_index>")
        sys.exit(1)
    
    try:
        patient_index = int(sys.argv[1])
    except ValueError:
        print("Error: patient_index must be an integer")
        sys.exit(1)
    
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
    print(f"Processing WhisperX for patient: {patient_id}")
    
    # Find all audio files for this patient
    audio_files = find_patient_audio_files(patient_dir)
    print(f"Found {len(audio_files)} audio files for patient {patient_id}")
    
    if len(audio_files) == 0:
        print(f"No audio files found for patient {patient_id}")
        sys.exit(0)
    
    # Process all files for this patient with WhisperX
    print(f"Starting WhisperX transcription for patient {patient_id}...")
    total_files = len(audio_files)
    successful = 0
    failed = 0
    skipped = 0
    
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n[{i}/{total_files}] Processing: {os.path.basename(audio_file)}")
        
        if check_existing_whisper_output(audio_file):
            print(f"  ↳ Skipping (WhisperX output exists)")
            skipped += 1
            continue
            
        success = transcribe_with_whisperx(audio_file)
        if success:
            successful += 1
        else:
            failed += 1
        
        print(f"  ↳ WhisperX Progress: {successful} successful, {failed} failed, {skipped} skipped")
    
    # Final summary
    print(f"\n" + "="*60)
    print(f"PATIENT {patient_id} WHISPERX PROCESSING COMPLETE")
    print(f"="*60)
    print(f"Total files: {total_files}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"="*60)
    
    if failed > 0:
        print("❌ Some WhisperX files failed to process")
        sys.exit(1)
    else:
        print("✓ All WhisperX files processed successfully")

if __name__ == "__main__":
    main()