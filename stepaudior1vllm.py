import base64
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

import requests
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

logger = logging.getLogger(__name__)


def _load_audio_segment(audio_path: str) -> AudioSegment:
    """Load audio via format-specific decoder with generic fallback."""
    file_ext = Path(audio_path).suffix.lower()
    loader = None
    if file_ext == ".mp3":
        loader = AudioSegment.from_mp3
    elif file_ext == ".wav":
        loader = AudioSegment.from_wav

    if loader is not None:
        try:
            return loader(audio_path)
        except CouldntDecodeError as err:
            logger.warning(
                "Primary decoder failed for %s (%s). Falling back to generic loader.",
                audio_path,
                err,
            )

    try:
        return AudioSegment.from_file(audio_path)
    except Exception as exc:  # noqa: BLE001
        logger.error("Generic loader failed for %s: %s", audio_path, exc)
        raise


class AudioService:
    """Audio processing utility that keeps the transformation reversible."""

    @staticmethod
    def read_audio_file(audio_path: str, max_duration: float = 29.9) -> Optional[List[bytes]]:
        """Read audio file and split into WAV chunks as bytes."""
        try:
            if not os.path.exists(audio_path):
                logger.error(f"Audio file not found: {audio_path}")
                return None

            file_size = os.path.getsize(audio_path)
            logger.info(f"Reading audio: {audio_path}, size: {file_size/1024:.2f} KB")

            if file_size == 0:
                logger.error(f"Audio file is empty: {audio_path}")
                return None

            audio = _load_audio_segment(audio_path)

            total_duration = len(audio) / 1000.0
            logger.info(f"Audio duration: {total_duration:.2f}s")

            audio_chunks: List[bytes] = []
            max_duration_ms = int(max_duration * 1000)

            if total_duration <= max_duration:
                audio_chunks.append(audio.export(format="wav").read())
            else:
                num_chunks = int(total_duration // max_duration) + 1
                logger.info(f"Splitting audio into {num_chunks} chunks")
                for i in range(num_chunks):
                    start_time = i * max_duration_ms
                    end_time = min((i + 1) * max_duration_ms, len(audio))
                    chunk = audio[start_time:end_time]
                    audio_chunks.append(chunk.export(format="wav").read())

            logger.info(f"Successfully processed {audio_path} into {len(audio_chunks)} chunk(s)")
            return audio_chunks

        except Exception as exc:
            logger.error(f"Failed to read audio file {audio_path}: {exc}", exc_info=True)
            return None

    @staticmethod
    def encode_audio_to_base64(audio_data: Union[bytes, List[bytes]]) -> List[str]:
        """Encode raw audio bytes to base64 strings."""
        if isinstance(audio_data, list):
            return [base64.b64encode(chunk).decode("utf-8") for chunk in audio_data]
        return [base64.b64encode(audio_data).decode("utf-8")]

    @staticmethod
    def validate_audio(audio_path: str) -> bool:
        """Ensure audio file exists and is readable."""
        try:
            if not os.path.exists(audio_path):
                return False

            file_size = os.path.getsize(audio_path)
            if file_size == 0:
                return False

            _ = _load_audio_segment(audio_path)
            return True
        except Exception as exc:
            logger.error(f"Audio validation failed for {audio_path}: {exc}")
            return False

    @staticmethod
    def get_audio_info(audio_path: str) -> dict:
        """Return basic audio metadata."""
        try:
            audio = _load_audio_segment(audio_path)
            return {
                "duration": len(audio) / 1000.0,
                "sample_rate": audio.frame_rate,
                "channels": audio.channels,
                "sample_width": audio.sample_width,
                "frame_count": audio.frame_count(),
            }
        except Exception as exc:
            logger.error(f"Failed to get audio info for {audio_path}: {exc}")
            return {}


class StepAudioR1:
    audio_token_re = re.compile(r"<audio_(\d+)>")

    def __init__(self, api_url, model_name, chat_template=None):
        self.api_url = api_url
        self.model_name = model_name
        # self.chat_template = chat_template or DEFAULT_CHAT_TEMPLATE
        self.log_dir = "request_logs"
        os.makedirs(self.log_dir, exist_ok=True)

    def __call__(self, messages, **kwargs):
        return next(self.stream(messages, **kwargs, stream=False))

    def log_request(self, payload):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(self.log_dir, f"request_{timestamp}.json")

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        return filename

    def stream(self, messages, stream=True, stop=None, **kwargs):
        headers = {"Content-Type": "application/json"}
        payload = kwargs
        payload["messages"] = self.apply_chat_template(messages)
        payload["model"] = self.model_name
        payload["stream"] = stream
        payload["skip_special_tokens"] = False

        if stop is None:
            stop = ["<|EOT|>"]

        if (
            payload["messages"][-1].get("role", None) == "assistant"
            and payload["messages"][-1].get("content", None) is None
        ):
            payload["messages"].pop(-1)
            payload["continue_final_message"] = False
            payload["add_generation_prompt"] = True
        elif payload["messages"][-1].get("eot", True):
            payload["continue_final_message"] = False
            payload["add_generation_prompt"] = True
        else:
            payload["continue_final_message"] = True
            payload["add_generation_prompt"] = False

        self.log_request(payload)
        
        with requests.post(self.api_url, headers=headers, json=payload, stream=stream) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if line == b'':
                    continue
                try:
                    # Handle SSE format: "data: {...}"
                    line_str = line.decode('utf-8')
                    if stream and line_str.startswith('data: '):
                        line_str = line_str[6:]  # Remove "data: " prefix

                    if line_str == '[DONE]':
                        break

                    # Parse JSON
                    data = json.loads(line_str)
                    choice_data = data['choices'][0]
                    line_content = choice_data['delta'] if stream else choice_data['message']

                    # print(choice_data)

                    # Extract text and audio
                    text = line_content.get('tts_content', {}).get('tts_text', None)
                    text = text if text is not None else line_content.get('content', '')

                    audio = line_content.get('tts_content', {}).get('tts_audio', None)
                    audio = [int(i) for i in StepAudioR1.audio_token_re.findall(audio)] if audio else None

                    yield line_content, text, audio

                except json.JSONDecodeError:
                    # Skip invalid JSON lines silently in streaming
                    continue
                except (KeyError, IndexError):
                    # Skip malformed response chunks
                    continue

    def process_content_item(self, item):
        if item["type"] != "audio":
            return [item]

        chunks = AudioService.read_audio_file(item["audio"], max_duration=25.0)
        if not chunks:
            logger.error(f"Failed to process audio item: {item['audio']}")
            return [item]

        encoded_chunks = AudioService.encode_audio_to_base64(chunks)
        
        return [
            {"type": "input_audio", "input_audio": {"data": chunk, "format": "wav"}}
            for chunk in encoded_chunks
        ]

    def apply_chat_template(self, messages):
        output_messages = []
        for message in messages:
            if message["role"] == "human" and isinstance(message["content"], list):
                processed = [j for i in message["content"] for j in self.process_content_item(i)]
                output_messages.append({"role": message["role"], "content": processed})
            else:
                output_messages.append(message)
        return output_messages
