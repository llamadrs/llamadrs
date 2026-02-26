#!/usr/bin/env python3
"""
DeepFilterNet3 audio processing script for CAMI project.
Processes audio tracks from MKV files using DeepFilterNet3 for noise reduction.
Handles large files by chunking them to avoid 32-bit indexing errors.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
import argparse
import time
import tempfile
import shutil
from typing import List, Tuple, Optional
from tqdm import tqdm
import glob

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class AudioProcessor:
    """Handle audio processing with DeepFilterNet3, including chunking for large files."""
    
    def __init__(self, chunk_duration: int = 60):
        self.chunk_duration = chunk_duration  # Default 60 seconds per chunk
        self.stats = {
            'total_files': 0,
            'processed': 0,
            'skipped': 0,
            'chunked': 0,
            'errors': []
        }
    
    def find_audio_files_from_mkv_paths(self, mkv_paths: List[str]) -> List[Path]:
        """Find all audio files to process from given MKV file paths."""
        audio_files = []
        
        for mkv_path in mkv_paths:
            mkv_path = mkv_path.strip()
            if not mkv_path:
                continue
                
            logger.info(f"Processing MKV path: {mkv_path}")
            
            # Get the directory containing the MKV file
            mkv_file = Path(mkv_path)
            if not mkv_file.exists():
                # Handle wildcard characters in filename
                # If the path contains wildcards, try to find matching files
                if '*' in mkv_path:
                    matching_files = glob.glob(mkv_path)
                    if matching_files:
                        mkv_file = Path(matching_files[0])  # Use first match
                    else:
                        logger.warning(f"No files found matching pattern: {mkv_path}")
                        continue
                else:
                    logger.warning(f"MKV file not found: {mkv_path}")
                    continue
            
            session_dir = mkv_file.parent
            audio_tracks_dir = session_dir / 'audio_tracks'
            
            if not audio_tracks_dir.exists():
                logger.warning(f"Audio tracks directory not found: {audio_tracks_dir}")
                continue
            
            logger.info(f"Checking audio tracks in: {audio_tracks_dir}")
            
            # Find all *_48k.wav files in the audio_tracks directory
            for audio_file_path in audio_tracks_dir.glob("*_48k.wav"):
                # Skip if already processed (check for DeepFilterNet3 version)
                output_name = audio_file_path.stem + "_DeepFilterNet3.wav"
                output_path = audio_file_path.parent / output_name
                
                if output_path.exists():
                    logger.info(f"Skipping already processed file: {audio_file_path.name}")
                    self.stats['skipped'] += 1
                else:
                    logger.info(f"Found file to process: {audio_file_path.name}")
                    audio_files.append(audio_file_path)
        
        return sorted(audio_files)
    
    def get_audio_duration(self, audio_file: Path) -> float:
        """Get duration of audio file in seconds using ffprobe."""
        try:
            cmd = [
                "ffprobe", "-v", "error", "-show_entries",
                "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
                str(audio_file)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return float(result.stdout.strip())
        except Exception as e:
            logger.error(f"Failed to get duration of {audio_file}: {e}")
            return 0.0
    
    def split_audio_file(self, audio_file: Path, temp_dir: Path) -> List[Path]:
        """Split audio file into chunks."""
        duration = self.get_audio_duration(audio_file)
        if duration == 0:
            return []
        
        chunks = []
        chunk_count = int(duration / self.chunk_duration) + 1
        
        logger.info(f"Splitting {audio_file.name} into {chunk_count} chunks of {self.chunk_duration}s")
        
        for i in range(chunk_count):
            start_time = i * self.chunk_duration
            chunk_file = temp_dir / f"chunk_{i:03d}.wav"
            
            cmd = [
                "ffmpeg", "-i", str(audio_file),
                "-ss", str(start_time),
                "-t", str(self.chunk_duration),
                "-c", "copy",
                "-y",
                str(chunk_file)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if result.returncode == 0 and chunk_file.exists():
                chunks.append(chunk_file)
            else:
                logger.error(f"Failed to create chunk {i}: {result.stderr}")
        
        return chunks
    
    def merge_audio_chunks(self, chunk_files: List[Path], output_file: Path) -> bool:
        """Merge processed audio chunks back into a single file."""
        if not chunk_files:
            return False
        
        # Create a list file for ffmpeg concat
        list_file = chunk_files[0].parent / "concat_list.txt"
        with open(list_file, 'w') as f:
            for chunk in sorted(chunk_files):
                f.write(f"file '{chunk.absolute()}'\n")
        
        cmd = [
            "ffmpeg", "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-c", "copy",
            "-y",
            str(output_file)
        ]
        
        logger.info(f"Merging {len(chunk_files)} chunks into {output_file.name}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        
        # Clean up list file
        list_file.unlink(missing_ok=True)
        
        if result.returncode != 0:
            logger.error(f"Failed to merge chunks: {result.stderr}")
            return False
        
        return output_file.exists() and output_file.stat().st_size > 1024
    
    def process_with_deepfilter(self, audio_file: Path, output_dir: Path) -> bool:
        """Process a single audio file with DeepFilterNet3."""
        try:
            cmd = ["deepFilter", str(audio_file), "--output-dir", str(output_dir)]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            
            if result.returncode != 0:
                if "canUse32BitIndexMath" in result.stderr:
                    return False  # Will trigger chunking
                else:
                    logger.error(f"DeepFilter error: {result.stderr}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to process with DeepFilter: {e}")
            return False
    
    def process_audio_file(self, audio_file: Path) -> List[str]:
        """Process a single audio file, using chunking if necessary."""
        errors = []
        output_name = audio_file.stem + "_DeepFilterNet3.wav"
        output_path = audio_file.parent / output_name
        
        # Get file size
        file_size_mb = audio_file.stat().st_size / (1024 * 1024)
        logger.info(f"Processing {audio_file.name} ({file_size_mb:.1f} MB)")
        
        # Try processing the whole file first
        if self.process_with_deepfilter(audio_file, audio_file.parent):
            self.stats['processed'] += 1
            return errors
        
        # If failed with indexing error, use chunking
        logger.info(f"Processing failed, trying with chunking...")
        self.stats['chunked'] += 1
        
        # Create temporary directory for chunks
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Split the audio file
            chunks = self.split_audio_file(audio_file, temp_path)
            if not chunks:
                error_msg = f"Failed to split {audio_file}"
                errors.append(error_msg)
                return errors
            
            # Process each chunk
            processed_chunks = []
            for i, chunk in enumerate(chunks):
                logger.info(f"Processing chunk {i+1}/{len(chunks)}")
                
                # Try with original chunk size
                if not self.process_with_deepfilter(chunk, temp_path):
                    # If still failing, try with even smaller duration
                    logger.warning(f"Chunk {i} failed, splitting further...")
                    
                    # Create sub-chunks
                    sub_temp_dir = temp_path / f"sub_chunks_{i}"
                    sub_temp_dir.mkdir(exist_ok=True)
                    
                    # Use smaller chunk size (10 seconds)
                    old_duration = self.chunk_duration
                    self.chunk_duration = 10
                    sub_chunks = self.split_audio_file(chunk, sub_temp_dir)
                    self.chunk_duration = old_duration
                    
                    # Process sub-chunks
                    processed_sub_chunks = []
                    for j, sub_chunk in enumerate(sub_chunks):
                        if self.process_with_deepfilter(sub_chunk, sub_temp_dir):
                            processed_sub = sub_temp_dir / f"{sub_chunk.stem}_DeepFilterNet3.wav"
                            if processed_sub.exists():
                                processed_sub_chunks.append(processed_sub)
                    
                    # Merge sub-chunks
                    if processed_sub_chunks:
                        merged_chunk = temp_path / f"chunk_{i:03d}_DeepFilterNet3.wav"
                        if self.merge_audio_chunks(processed_sub_chunks, merged_chunk):
                            processed_chunks.append(merged_chunk)
                        else:
                            error_msg = f"Failed to merge sub-chunks for chunk {i}"
                            errors.append(error_msg)
                else:
                    # Chunk processed successfully
                    processed_chunk = temp_path / f"{chunk.stem}_DeepFilterNet3.wav"
                    if processed_chunk.exists():
                        processed_chunks.append(processed_chunk)
            
            # Merge all processed chunks
            if processed_chunks:
                if self.merge_audio_chunks(processed_chunks, output_path):
                    logger.info(f"Successfully created {output_path.name}")
                    self.stats['processed'] += 1
                else:
                    error_msg = f"Failed to merge final output for {audio_file}"
                    errors.append(error_msg)
            else:
                error_msg = f"No chunks were successfully processed for {audio_file}"
                errors.append(error_msg)
        
        return errors
    
    def run(self, mkv_paths: List[str]):
        """Run the complete processing pipeline for given MKV paths."""
        # Find all audio files to process from the MKV paths
        logger.info(f"Processing {len(mkv_paths)} MKV paths")
        audio_files = self.find_audio_files_from_mkv_paths(mkv_paths)
        self.stats['total_files'] = len(audio_files)
        
        if self.stats['total_files'] == 0:
            logger.info("No audio files found to process (all already processed or no valid paths)")
            return
        
        logger.info(f"Found {self.stats['total_files']} files to process")
        
        # Process each file
        start_time = time.time()
        
        for i, audio_file in enumerate(audio_files, 1):
            logger.info(f"\n[{i}/{self.stats['total_files']}] Processing {audio_file}")
            errors = self.process_audio_file(audio_file)
            self.stats['errors'].extend(errors)
        
        # Final summary
        elapsed_time = time.time() - start_time
        self.print_summary(elapsed_time)
    
    def print_summary(self, elapsed_time: float):
        """Print processing summary."""
        logger.info("\n" + "="*60)
        logger.info("PROCESSING SUMMARY")
        logger.info("="*60)
        logger.info(f"Total audio files found: {self.stats['total_files']}")
        logger.info(f"Files processed: {self.stats['processed']}")
        logger.info(f"Files skipped (already processed): {self.stats['skipped']}")
        logger.info(f"Files requiring chunking: {self.stats['chunked']}")
        logger.info(f"Errors encountered: {len(self.stats['errors'])}")
        logger.info(f"Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.1f} minutes)")
        
        if self.stats['errors']:
            logger.error("\nError details:")
            for error in self.stats['errors']:
                logger.error(f"  - {error}")
        else:
            logger.info("\nAll files processed successfully!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Process CAMI audio files with DeepFilterNet3 (with automatic chunking)"
    )
    parser.add_argument(
        "mkv_paths",
        nargs="+",
        help="MKV file paths to process (can include wildcards)"
    )
    parser.add_argument(
        "--chunk-duration",
        type=int,
        default=60*30,
        help="Duration of audio chunks in seconds (default: 1800 = 30 minutes)"
    )
    parser.add_argument(
        "--conda-env",
        default="deepfilter",
        help="Conda environment name (default: deepfilter)"
    )
    
    args = parser.parse_args()
    
    # Verify conda environment
    current_env = os.environ.get('CONDA_DEFAULT_ENV', '')
    if current_env != args.conda_env:
        logger.warning(f"Current conda environment: {current_env}, expected: {args.conda_env}")
        logger.info("Continuing anyway, but make sure DeepFilterNet3 is available")
    
    # Log the input paths
    logger.info(f"Received {len(args.mkv_paths)} MKV paths:")
    for path in args.mkv_paths:
        logger.info(f"  - {path}")
    
    # Create and run processor
    processor = AudioProcessor(args.chunk_duration)
    processor.run(args.mkv_paths)


if __name__ == "__main__":
    main()