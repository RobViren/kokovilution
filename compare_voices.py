import os
import numpy as np
import soundfile as sf
from kokoro_onnx import Kokoro
from pathlib import Path
import argparse

def load_voice_embedding(file_path: str) -> np.ndarray:
    """Load a voice embedding from a .bin file"""
    with open(file_path, 'rb') as f:
        data = np.load(f)
        return data['evolved_voice']

def generate_samples(
    input_dir: str,
    output_dir: str,
    text: str,
    kokoro_path: str = "kokoro-v1.0.onnx",
    voices_path: str = "voices-v1.0.bin"
):
    """Generate audio samples for all voice models in the input directory"""
    # Initialize Kokoro
    kokoro = Kokoro(kokoro_path, voices_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all .bin files in input directory
    voice_files = list(Path(input_dir).glob("*.bin"))
    
    print(f"Found {len(voice_files)} voice models")
    print(f"Generating audio for text: '{text}'")
    
    # Process each voice file
    for voice_file in voice_files:
        try:
            # Load the voice embedding
            voice_embedding = load_voice_embedding(str(voice_file))
            
            # Generate audio
            audio, sr = kokoro.create(text, voice=voice_embedding)
            
            # Create output filename
            output_filename = f"{voice_file.stem}_{text[:30]}.wav"
            output_path = os.path.join(output_dir, output_filename)
            
            # Save audio
            sf.write(output_path, audio, sr)
            print(f"Generated: {output_filename}")
            
        except Exception as e:
            print(f"Error processing {voice_file.name}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Generate audio samples from evolved voice models")
    parser.add_argument("--input-dir", required=True, help="Directory containing .bin voice models")
    parser.add_argument("--output-dir", required=True, help="Directory to save generated audio files")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--kokoro-path", default="kokoro-v1.0.onnx", help="Path to Kokoro model")
    parser.add_argument("--voices-path", default="voices-v1.0.bin", help="Path to voices file")
    
    args = parser.parse_args()
    
    generate_samples(
        args.input_dir,
        args.output_dir,
        args.text,
        args.kokoro_path,
        args.voices_path
    )

if __name__ == "__main__":
    main() 