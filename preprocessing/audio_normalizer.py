#!/usr/bin/env python3
"""
Audio Loudness Normalizer for CAMI Mental Health Interview Videos

This script normalizes the loudness of all DeepFilterNet3 processed audio files
to a consistent, loud level using FFmpeg's loudnorm filter (EBU R128 standard).
"""

import os
import subprocess
import glob
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_audio_files(root_directory, pattern="*_DeepFilterNet3.wav"):
    """
    Find all DeepFilterNet3 processed audio files in the directory structure.
    
    Args:
        root_directory (str): Root directory to search
        pattern (str): File pattern to match
    
    Returns:
        list: List of audio file paths
    """
    audio_files = []
    search_pattern = os.path.join(root_directory, "**", pattern)
    audio_files = glob.glob(search_pattern, recursive=True)
    
    audio_files = [f for f in audio_files if os.path.isfile(f) and not os.path.isfile(f.replace("DeepFilterNet3", "DeepFilterNet3_normalized"))]
    
    logger.info(f"Found {len(audio_files)} audio files matching pattern '{pattern}'")
    return sorted(audio_files)

def normalize_loudness_ffmpeg(input_file, output_file, target_lufs=-16, target_peak=-1.0, sample_rate=48000, use_local_tmp=False):
    """
    Normalize audio loudness using FFmpeg's loudnorm filter.
    
    Args:
        input_file (str): Input audio file path
        output_file (str): Output audio file path
        target_lufs (float): Target loudness in LUFS (-23 to -16 typical, -16 is quite loud)
        target_peak (float): Target peak level in dBFS
        sample_rate (int): Output sample rate
        use_local_tmp (bool): Use local node storage for processing
    
    Returns:
        bool: True if successful, False otherwise
    """
    import tempfile
    import shutil
    import time
    
    try:
        if use_local_tmp and os.environ.get('TMPDIR'):
            # Use local node storage if available
            local_tmp_dir = os.environ.get('TMPDIR', '/tmp')
            temp_input = os.path.join(local_tmp_dir, f"tmp_input_{os.getpid()}_{int(time.time())}.wav")
            temp_output = os.path.join(local_tmp_dir, f"tmp_output_{os.getpid()}_{int(time.time())}.wav")
            # Ensure temp directory exists
            os.makedirs(local_tmp_dir, exist_ok=True)
            # Copy input to local storage
            shutil.copy2(input_file, temp_input)
            process_input = temp_input
            process_output = temp_output
        else:
            process_input = input_file
            process_output = output_file
        
        # FFmpeg command with I/O optimizations
        cmd = [
            'ffmpeg',
            '-i', process_input,
            '-af', f'loudnorm=I={target_lufs}:TP={target_peak}:LRA=7:print_format=summary',
            '-ar', str(sample_rate),
            '-threads', '1',  # Limit threads per process
            '-y',  # Overwrite output file if it exists
            process_output
        ]
        
        # Run FFmpeg
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Copy output back if using local tmp
        if use_local_tmp and os.environ.get('TMPDIR'):
            shutil.copy2(temp_output, output_file)
            # Clean up temp files
            try:
                os.unlink(temp_input)
                os.unlink(temp_output)
            except:
                pass
        
        # Log the loudnorm statistics if available
        if "Input Integrated:" in result.stderr:
            logger.info(f"Processed: {os.path.basename(input_file)}")
            # Extract and log key statistics
            lines = result.stderr.split('\n')
            for line in lines:
                if any(stat in line for stat in ["Input Integrated:", "Output Integrated:", "Target Offset:"]):
                    logger.debug(f"  {line.strip()}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error processing {input_file}: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.error("FFmpeg not found. Please install FFmpeg and ensure it's in your PATH.")
        return False
    except Exception as e:
        logger.error(f"Unexpected error processing {input_file}: {str(e)}")
        return False

def create_output_path(input_file, suffix="_normalized"):
    """
    Create output file path by adding suffix before file extension.
    
    Args:
        input_file (str): Input file path
        suffix (str): Suffix to add to filename
    
    Returns:
        str: Output file path
    """
    path = Path(input_file)
    output_file = path.parent / f"{path.stem}{suffix}{path.suffix}"
    return str(output_file)

def process_single_file(input_file, target_lufs=-16, suffix="_normalized", use_local_tmp=False):
    """
    Process a single audio file for loudness normalization.
    
    Args:
        input_file (str): Input audio file path
        target_lufs (float): Target loudness level
        suffix (str): Output file suffix
        use_local_tmp (bool): Use local node storage
    
    Returns:
        tuple: (input_file, success_boolean)
    """
    output_file = create_output_path(input_file, suffix)
    success = normalize_loudness_ffmpeg(input_file, output_file, target_lufs=target_lufs, use_local_tmp=use_local_tmp)
    return (input_file, success)

def main():
    parser = argparse.ArgumentParser(description="Normalize loudness of CAMI audio files")
    parser.add_argument("--root_directory", help="Root directory containing CAMI audio files", default="/home/gyk/CAMI")
    parser.add_argument("--target-lufs", type=float, default=-12, 
                       help="Target loudness in LUFS (-23 to -12, lower is quieter). Default: -16 (quite loud)")
    parser.add_argument("--suffix", default="_normalized", 
                       help="Suffix for output files. Default: '_normalized'")
    parser.add_argument("--max-workers", type=int, default=2, 
                       help="Maximum number of parallel processes. Default: 2 (cluster-friendly)")
    parser.add_argument("--pattern", default="/home/gyk/CAMI/D6GMM/D6GMM_211210_17_45_14/audio_tracks/2021-12-10 17-45-14_audio_track1_48k_DeepFilterNet3.wav",
                       help="File pattern to match. Default: '*_DeepFilterNet3.wav'")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be processed without actually processing")
    parser.add_argument("--use-local-tmp", action="store_true",
                       help="Use local node storage for temporary processing")
    parser.add_argument("--batch-delay", type=float, default=0.1,
                       help="Delay between starting processes (seconds). Default: 0.1")
    
    args = parser.parse_args()
    
    # Validate target LUFS
    if args.target_lufs < -30 or args.target_lufs > -6:
        logger.warning(f"Target LUFS {args.target_lufs} is outside typical range (-30 to -6)")
    
    # Find all audio files
    audio_files = find_audio_files(args.root_directory, args.pattern)
    
    if not audio_files:
        logger.error(f"No audio files found in {args.root_directory} matching pattern '{args.pattern}'")
        return
    
    logger.info(f"Target loudness: {args.target_lufs} LUFS")
    logger.info(f"Output suffix: {args.suffix}")
    
    if args.dry_run:
        logger.info("DRY RUN - Files that would be processed:")
        for file in audio_files:
            output_file = create_output_path(file, args.suffix)
            print(f"  {file} -> {output_file}")
        return
    
    # Process files with staggered start and limited concurrency
    successful = 0
    failed = 0
    
    logger.info(f"Starting processing with {args.max_workers} workers (cluster-friendly)...")
    logger.info(f"Batch delay: {args.batch_delay}s between process starts")
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit tasks with staggered timing to reduce I/O spikes
        future_to_file = {}
        for i, file in enumerate(audio_files):
            if i > 0 and args.batch_delay > 0:
                time.sleep(args.batch_delay)  # Stagger process starts
            
            future = executor.submit(process_single_file, file, args.target_lufs, args.suffix, args.use_local_tmp)
            future_to_file[future] = file
        
        # Process completed tasks
        for future in as_completed(future_to_file):
            input_file, success = future.result()
            if success:
                successful += 1
                logger.info(f"✓ Completed: {os.path.basename(input_file)} ({successful}/{len(audio_files)})")
            else:
                failed += 1
                logger.error(f"✗ Failed: {input_file}")
    
    # Summary
    logger.info(f"\nProcessing complete!")
    logger.info(f"Successfully processed: {successful} files")
    logger.info(f"Failed: {failed} files")
    logger.info(f"Total: {len(audio_files)} files")

if __name__ == "__main__":
    main()

# Example usage:
# Cluster-friendly with local temp storage:
# python audio_normalizer.py /home/gyk/CAMI --max-workers 2 --use-local-tmp
# 
# Conservative approach:
# python audio_normalizer.py /home/gyk/CAMI --max-workers 1
#
# SLURM job array approach (recommended for large datasets):
# sbatch --array=1-100 process_audio_array.sh