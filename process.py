import os
import argparse
import logging
from faster_whisper_tools.transcribe_audio import transcribe_audio_and_save_to_file, AVAILABLE_MODELS


def transcribe_all_audio(audio_dir: str, model_size: str, output_dir: str):
    """Walks the given audio_dir, transcribing with the model_size and saving them to the output_dir"""
    output_files = []
    for root, _, files in os.walk(audio_dir):
        for file in files:
            if file.endswith((".mp3", ".wav", ".m4a", ".ogg")):
                audio_file_path = os.path.join(root, file)
                try:
                    output_file = transcribe_audio_and_save_to_file(audio_file_path, model_size, output_dir)
                    output_files.append(output_file)
                except Exception as e:
                    logging.error(f"Error transcribing {audio_file_path}: {e}")
    return output_files


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Transcribe all audio files in a directory.")
    parser.add_argument("--audio_dir", type=str, required=True, help="Directory containing audio files")
    parser.add_argument("--model", type=str, choices=AVAILABLE_MODELS, default="tiny.en", help="Model size for transcription (default: tiny.en)")
    parser.add_argument("--output-dir", type=str, default="./output", help="Directory to save transcription output (default: './output')")
    args = parser.parse_args()

    output_files = transcribe_all_audio(args.audio_dir, args.model, args.output_dir)
    logging.info(f"Transcriptions saved to: {[f for f in output_files]}")


if __name__ == "__main__":
    main()
