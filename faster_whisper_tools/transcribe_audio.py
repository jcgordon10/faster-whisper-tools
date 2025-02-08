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
    """
    A class representing a subtitle segment in SRT format.

    Attributes:
        index (int): The sequence number of the subtitle segment
        start_time (float): The start time of the subtitle in seconds
        end_time (float): The end time of the subtitle in seconds
        text (str): The text content of the subtitle

    Methods:
        to_string(): Converts the segment to properly formatted SRT string
        _format_timestamp(): Static method to format time in HH:MM:SS,mmm format
    """

    def __init__(self, index: int, start_time: float, end_time: float, text: str):
        self.index = index
        self.start_time = start_time
        self.end_time = end_time
        self.text = text

    def to_string(self) -> str:
        """
        Returns a string in the format:
            1
            00:00:00,000 --> 00:00:04,799
            This is an example sentece from the transcription
        """
        start_timestamp = self._format_timestamp(self.start_time)
        end_timestamp = self._format_timestamp(self.end_time)
        return f"{self.index}\n{start_timestamp} --> {end_timestamp}\n{self.text}\n\n"

    @staticmethod
    def _format_timestamp(time: float) -> str:
        """
        converts time from seconds (float), e.g. `4.8` to HH:MM:SS,mmm format (str), e.g. `00:00:04,799`
        """
        return "{:02d}:{:02d}:{:02d},{:03d}".format(
            int(time // 3600),
            int((time % 3600) // 60),
            int(time % 60),
            int((time % 1) * 1000),
        )


def transcribe_audio(audio_file_path: str, model_size: str) -> Tuple[str, List[SRTSegment]]:
    """
    Transcribe an audio file using the specified model size and return the transcription and SRT segments.

    Args:
        audio_file_path (str): The path to the audio file to be transcribed.
        model_size (str): The size of the model to use for transcription.

    Returns:
        Tuple[str, List[SRTSegment]]: A tuple containing the full transcription as a string and a list of SRTSegment objects.
    """
    model = WhisperModel(model_size, device="cuda", compute_type="int8")
    segments, info = model.transcribe(audio_file_path, beam_size=5)

    full_txt = []
    srt_segments = []
    start_time = 0.0

    # nice clean steady tqdm bar
    with tqdm(total=info.duration, bar_format="{l_bar}{bar}| {n:.0f}/{total:.0f} [{elapsed}<{remaining}, {rate_fmt}{postfix}]", unit=" audio seconds") as pbar:
        for i, segment in enumerate(segments, start=1):
            full_txt.append(segment.text.strip())
            srt_segments.append(SRTSegment(i, start_time, segment.end, segment.text.strip()))
            pbar.update(segment.end - start_time)
            start_time = segment.end
    return " ".join(full_txt), srt_segments


def save_transcription(transcription: str, output_dir: str) -> None:
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_text_file = os.path.join(output_dir, f"{now_str}_output.txt")
    os.makedirs(os.path.dirname(output_text_file), exist_ok=True)
    with open(output_text_file, "w", encoding="utf-8") as file:
        file.write(transcription)
    logging.info(f"Transcription saved to {output_text_file}")


def save_srt(srt_segments: List[SRTSegment], output_dir: str) -> None:
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_srt_file = os.path.join(output_dir, f"{now_str}_output.srt")
    with open(output_srt_file, "w", encoding="utf-8") as file:
        for segment in srt_segments:
            file.write(segment.to_string())
    logging.info(f"SRT file saved to {output_srt_file}")


def transcribe_audio_and_save_to_file(audio_file_path: str, model_size: str, output_dir: str, captions: bool = False) -> None:
    transcription, srt_segments = transcribe_audio(audio_file_path, model_size)
    save_transcription(transcription, output_dir)
    if captions:
        save_srt(srt_segments, output_dir)
    return transcription, srt_segments


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Transcribe audio from a local audio file.")
    parser.add_argument("--filename", type=str, required=True, help="Local audio file path")
    parser.add_argument("--model", type=str, choices=AVAILABLE_MODELS, default="tiny.en", help="Model size for transcription (default: tiny.en)")
    parser.add_argument("--output-dir", type=str, default="./output", help="Directory to save transcription output (default: './output')")
    args = parser.parse_args()

    transcription, srt_segments = transcribe_audio_and_save_to_file(args.filename, args.model, args.output_dir)
    print(transcription)
    # for segment in srt_segments:
    #     print(segment.to_string()) # 1\n 00:00:00,000 --> 00:00:04,799\n This is an example sentence
    #     print(segment.index, segment.start_time, segment.end_time, segment.text) # 1 0.0 4.8 This is an example sentence


if __name__ == "__main__":
    main()
