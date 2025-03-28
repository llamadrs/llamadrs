#!/bin/bash
# whisper_transcription.sh
# This script processes audio files in a specified directory using whisperx for transcription.
# Usage: ./whisper_transcription.sh <directory_path>

source ~/anaconda3/etc/profile.d/conda.sh
conda activate whisperx

export CUDA_VISIBLE_DEVICES=0

# Define CAMI directory path
SRC=$1

# Function to process a single file
process_file() {
    input_file="$1"
    src_dir=$(dirname "$input_file")
    
    # Get the base filename without extension
    filename=$(basename "$input_file")
    base_name="${filename%.*}"
    
    # Set output paths
    input_file="$src_dir/${base_name}.wav"
    
    # Check if VTT file exists
    vtt_file="${src_dir}/${base_name}.vtt"
    if [[ -f "$vtt_file" ]]; then
        echo "VTT file found: $vtt_file. Skipping..."
        exit
    fi

    echo "Processing: $filename"
        
        # Run whisperx transcription
        echo "Transcribing with whisperx..."
        whisperx "$input_file" \
            --model large \
            --language en \
            --output_dir "$src_dir"
        echo "✓ Successfully transcribed: $base_name"
    
    echo "----------------------------------------"
}




# Find all WAV files in the CAMI directory
mapfile -d '' FILES < <(find "$SRC" -name "*.wav" -print0)
echo "Found ${#FILES[@]} files to process."

# Process files in parallel using background jobs
input_file="${FILES[$1]}"
process_file "$input_file"