import argparse
import logging
import os
from datetime import datetime
from typing import Tuple, List
from tqdm import tqdm
from faster_whisper import WhisperModel

AVAILABLE_MODELS = [
    "tiny.en",
    "tiny",
    "base.en",
    "base",
    "small.en",
    "small",
    "medium.en",
    "medium",
    "large",
    "large-v2",
    "large-v3",
]

class SRTSegment:
    def __init__(self, index: int, start_time: float, end_time: float, text: str):
        self.index = index
        self.start_time = start_time
        self.end_time = end_time
        self.text = text

    def to_srt_string(self) -> str:
        start_timestamp = self._format_timestamp(self.start_time)
        end_timestamp = self._format_timestamp(self.end_time)
        return f"{self.index}\n{start_timestamp} --> {end_timestamp}\n{self.text}\n"

    @staticmethod
    def _format_timestamp(time: float) -> str:
        return "{:02d}:{:02d}:{:02d},{:03d}".format(
            int(time // 3600),
            int((time % 3600) // 60),
            int(time % 60),
            int((time % 1) * 1000),
        )


def transcribe_audio(audio_file_path: str, model_size: str) -> Tuple[str, List[SRTSegment]]:
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    segments, info = model.transcribe(audio_file_path, beam_size=5)

    full_txt = []
    srt_segments = []
    start_time = 0.0

    with tqdm(total=info.duration, unit=" audio seconds") as pbar:
        for idx, segment in enumerate(segments, start=1):
            full_txt.append(segment.text)
            srt_segment = SRTSegment(idx, start_time, segment.end, segment.text)
            srt_segments.append(srt_segment)
            pbar.update(segment.end - start_time)
            start_time = segment.end

    return " ".join(full_txt), srt_segments


def save_to_file(text: str, output_file: str):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(text)


def save_transcription_and_summarize(transcription: str, output_dir: str):
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_text_file = os.path.join(output_dir, f'{now_str}_output.txt')
    save_to_file(transcription, output_text_file)
    logging.info(f"Transcription saved to {output_text_file}")
    return output_text_file


def process_audio_file(audio_file_path: str, model_size: str, output_dir: str):
    transcription, srt_segments = transcribe_audio(audio_file_path, model_size, output_dir)
    output_text_file = save_transcription_and_summarize(transcription, output_dir)
    return output_text_file


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Transcribe audio from a local audio file."
    )
    parser.add_argument("--filename", type=str, required=True, help="Local audio file path")
    parser.add_argument(
        "--model",
        type=str,
        choices=AVAILABLE_MODELS,
        default="tiny.en",
        help="Model size for transcription (default: tiny.en)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default='./output',
        help="Directory to save transcription output (default: './output')",
    )
    args = parser.parse_args()

    process_audio_file(args.filename, args.model, args.output_dir)


if __name__ == "__main__":
    main()
