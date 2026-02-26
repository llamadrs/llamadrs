#!/bin/bash
# get command line arguments as list of directories
source ~/anaconda3/etc/profile.d/conda.sh
conda activate deepfilter
FILES=("$@")
# make it an array
FILES=($FILES)
total_files=${#FILES[@]}
echo "Found $total_files files to process."
current_file=0
for input_dir in "${FILES[@]}"; do
    input_files_str="$(find $input_dir/*/*/*.mkv -type f ! -name "*left*" ! -name "*right*" ! -name ".*" -name "* *")"
    IFS=$'\n' read -r -d '' -a input_files <<< "$input_files_str"
    for input_file in "${input_files[@]}"; do
        # remove newline at the end of the string
        input_file=$(echo $input_file | tr -d '\n')
        # Extract audio tracks as WAV files
        last_name=$(basename "$input_file" ".mkv")
        dir_name=$(dirname "$input_file")/audio_tracks
        
        # Create the output directory if it doesn't exist
        mkdir -p "$dir_name"
        
        output_file="$dir_name"/"$last_name"_audio_track1_48k_DeepFilterNet3.wav
        if [[ ! -f "$output_file" ]]; then
            echo "Extracting track 1 from $input_file"
            ffmpeg -i "$input_file" -ac 1 -map 0:a:0 -c:a pcm_s16le -threads 64 -ar 48000 -y "$dir_name"/"$last_name"_audio_track1_48k.wav
        fi

        output_file="$dir_name"/"$last_name"_audio_track2_48k_DeepFilterNet3.wav
        if [[ ! -f "$output_file" ]]; then
            echo "Extracting track 2 from $input_file"
            ffmpeg -i "$input_file" -ac 1 -map 0:a:1 -c:a pcm_s16le -threads 64 -ar 48000 -y "$dir_name"/"$last_name"_audio_track2_48k.wav
        fi

        output_file="$dir_name"/"$last_name"_audio_track3_48k_DeepFilterNet3.wav
        if [[ ! -f "$output_file" ]]; then
            echo "Extracting track 3 from $input_file"
            ffmpeg -i "$input_file" -ac 1 -map 0:a:2 -c:a pcm_s16le -threads 64 -ar 48000 -y "$dir_name"/"$last_name"_audio_track3_48k.wav
        fi

        output_file="$dir_name"/"$last_name"_audio_track4_48k_DeepFilterNet3.wav
        if [[ ! -f "$output_file" ]]; then
            echo "Extracting track 4 from $input_file"
            ffmpeg -i "$input_file" -ac 1 -map 0:a:3 -c:a pcm_s16le -threads 64 -ar 48000 -y "$dir_name"/"$last_name"_audio_track4_48k.wav
        fi
    
        current_file=$((current_file + 1))
        echo "Processed $current_file of $total_files files."
    done
done