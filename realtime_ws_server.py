import argparse
import asyncio
import importlib.util
import json
import math
import pathlib
import threading
import time
from types import SimpleNamespace

import librosa
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import websockets


SUPPORTED_SAMPLE_RATES = {8000, 16000}
SUPPORTED_FORMAT = "pcm_s16le"


def select_device(gpu: int = 0) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu}" if gpu is not None else "cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_realtime_gui_module(device: torch.device):
    module_path = pathlib.Path(__file__).with_name("real-time-gui.py")
    spec = importlib.util.spec_from_file_location("seed_vc_realtime_gui", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.device = device
    return module


def pcm_s16le_to_float32(data: bytes, channels: int) -> np.ndarray:
    if len(data) % 2 != 0:
        data = data[:-1]
    samples = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        usable = (len(samples) // channels) * channels
        samples = samples[:usable].reshape(-1, channels).mean(axis=1)
    return np.clip(samples, -1.0, 1.0)


def float32_to_pcm_s16le(wav: np.ndarray) -> bytes:
    wav = np.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)
    wav = np.clip(wav, -1.0, 1.0)
    return (wav * 32767.0).astype("<i2").tobytes()


class RealtimePcmSession:
    def __init__(
        self,
        *,
        rt_module,
        model_set,
        device: torch.device,
        reference_audio_path: str,
        sample_rate: int,
        channels: int = 1,
        block_time: float = 0.18,
        crossfade_time: float = 0.04,
        extra_time_ce: float = 2.5,
        extra_time: float = 0.5,
        extra_time_right: float = 0.02,
        diffusion_steps: int = 6,
        inference_cfg_rate: float = 0.7,
        max_prompt_length: float = 3.0,
    ):
        if sample_rate not in SUPPORTED_SAMPLE_RATES:
            raise ValueError("sample_rate must be 8000 or 16000")
        if channels < 1:
            raise ValueError("channels must be >= 1")
        if extra_time_ce < extra_time:
            raise ValueError("extra_time_ce must be >= extra_time")

        self.rt_module = rt_module
        self.model_set = model_set
        self.device = device
        self.reference_audio_path = reference_audio_path
        self.sample_rate = sample_rate
        self.channels = channels
        self.block_time = block_time
        self.crossfade_time = crossfade_time
        self.extra_time_ce = extra_time_ce
        self.extra_time = extra_time
        self.extra_time_right = extra_time_right
        self.diffusion_steps = diffusion_steps
        self.inference_cfg_rate = inference_cfg_rate
        self.max_prompt_length = max_prompt_length

        mel_fn_args = model_set[-1]
        self.model_sample_rate = mel_fn_args["sampling_rate"]

        self.reference_wav, _ = librosa.load(reference_audio_path, sr=self.model_sample_rate)
        if self.reference_wav.size == 0:
            raise ValueError("reference audio is empty")

        self.zc = self.sample_rate // 50
        self.block_frame = max(
            self.zc,
            int(round(self.block_time * self.sample_rate / self.zc)) * self.zc,
        )
        self.block_frame_16k = 320 * self.block_frame // self.zc
        self.crossfade_frame = max(
            self.zc,
            int(round(self.crossfade_time * self.sample_rate / self.zc)) * self.zc,
        )
        self.sola_buffer_frame = min(self.crossfade_frame, 4 * self.zc)
        self.sola_search_frame = self.zc
        self.extra_frame = (
            int(round(self.extra_time_ce * self.sample_rate / self.zc)) * self.zc
        )
        self.extra_frame_right = (
            int(round(self.extra_time_right * self.sample_rate / self.zc)) * self.zc
        )

        input_len = (
            self.extra_frame
            + self.crossfade_frame
            + self.sola_search_frame
            + self.block_frame
            + self.extra_frame_right
        )
        self.input_wav = torch.zeros(input_len, device=device, dtype=torch.float32)
        self.input_wav_res = torch.zeros(320 * input_len // self.zc, device=device, dtype=torch.float32)
        self.sola_buffer = torch.zeros(self.sola_buffer_frame, device=device, dtype=torch.float32)
        self.skip_head = self.extra_frame // self.zc
        self.skip_tail = self.extra_frame_right // self.zc
        self.return_length = (
            self.block_frame + self.sola_buffer_frame + self.sola_search_frame
        ) // self.zc
        self.fade_in_window = (
            torch.sin(
                0.5
                * math.pi
                * torch.linspace(
                    0.0,
                    1.0,
                    steps=self.sola_buffer_frame,
                    device=device,
                    dtype=torch.float32,
                )
            )
            ** 2
        )
        self.fade_out_window = 1 - self.fade_in_window
        self.input_buffer = np.empty(0, dtype=np.float32)

        if self.model_sample_rate != self.sample_rate:
            self.output_resampler = torchaudio.transforms.Resample(
                orig_freq=self.model_sample_rate,
                new_freq=self.sample_rate,
                dtype=torch.float32,
            ).to(device)
        else:
            self.output_resampler = None

    @property
    def algorithm_delay_ms(self) -> int:
        return int(round((self.block_time * 2 + self.extra_time_right) * 1000))

    def push_pcm(self, payload: bytes, inference_lock: threading.Lock) -> list[bytes]:
        input_samples = pcm_s16le_to_float32(payload, self.channels)
        if input_samples.size == 0:
            return []
        self.input_buffer = np.concatenate([self.input_buffer, input_samples])

        outputs = []
        while self.input_buffer.size >= self.block_frame:
            block = self.input_buffer[: self.block_frame]
            self.input_buffer = self.input_buffer[self.block_frame :]
            outputs.append(self.process_block(block, inference_lock))
        return outputs

    def process_block(self, block: np.ndarray, inference_lock: threading.Lock) -> bytes:
        started_at = time.perf_counter()

        self.input_wav[: -self.block_frame] = self.input_wav[self.block_frame :].clone()
        self.input_wav[-self.block_frame :] = torch.from_numpy(block).to(self.device)

        self.input_wav_res[: -self.block_frame_16k] = self.input_wav_res[
            self.block_frame_16k :
        ].clone()
        resample_source = self.input_wav[-self.block_frame - 2 * self.zc :].detach().cpu().numpy()
        resampled = librosa.resample(
            resample_source,
            orig_sr=self.sample_rate,
            target_sr=16000,
        )[320:]
        resampled_tensor = torch.from_numpy(resampled).to(self.device, dtype=torch.float32)
        target_len = min(resampled_tensor.numel(), self.input_wav_res.numel())
        self.input_wav_res[-target_len:] = resampled_tensor[-target_len:]

        with inference_lock:
            infer_wav = self.rt_module.custom_infer(
                self.model_set,
                self.reference_wav,
                self.reference_audio_path,
                self.input_wav_res,
                self.block_frame_16k,
                self.skip_head,
                self.skip_tail,
                self.return_length,
                int(self.diffusion_steps),
                self.inference_cfg_rate,
                self.max_prompt_length,
                self.extra_time_ce - self.extra_time,
            )

        if self.output_resampler is not None:
            infer_wav = self.output_resampler(infer_wav)

        conv_input = infer_wav[None, None, : self.sola_buffer_frame + self.sola_search_frame]
        cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
        cor_den = torch.sqrt(
            F.conv1d(
                conv_input**2,
                torch.ones(1, 1, self.sola_buffer_frame, device=self.device),
            )
            + 1e-8
        )
        tensor = cor_nom[0, 0] / cor_den[0, 0]
        sola_offset = torch.argmax(tensor, dim=0).item() if tensor.numel() > 1 else int(tensor.item())

        infer_wav = infer_wav[sola_offset:]
        infer_wav[: self.sola_buffer_frame] *= self.fade_in_window
        infer_wav[: self.sola_buffer_frame] += self.sola_buffer * self.fade_out_window
        self.sola_buffer[:] = infer_wav[
            self.block_frame : self.block_frame + self.sola_buffer_frame
        ]

        output = infer_wav[: self.block_frame].detach().cpu().numpy()
        elapsed_ms = int(round((time.perf_counter() - started_at) * 1000))
        print(
            f"processed block: sr={self.sample_rate} frames={self.block_frame} "
            f"diffusion_steps={self.diffusion_steps} elapsed_ms={elapsed_ms}"
        )
        return float32_to_pcm_s16le(output)


class SeedVcWebSocketServer:
    def __init__(self, args):
        self.args = args
        self.device = select_device(args.gpu)
        print(f"Using device: {self.device}")
        self.rt_module = load_realtime_gui_module(self.device)
        self.rt_module.device = self.device
        self.rt_module.fp16 = args.fp16
        model_args = SimpleNamespace(
            checkpoint_path=args.checkpoint_path,
            config_path=args.config_path,
            fp16=args.fp16,
        )
        self.model_set = self.rt_module.load_models(model_args)
        self.inference_lock = threading.Lock()

    async def handler(self, websocket):
        session = None
        try:
            raw_config = await websocket.recv()
            if isinstance(raw_config, bytes):
                raise ValueError("first websocket message must be a JSON text config")
            config = json.loads(raw_config)
            session = self.create_session(config)
            await websocket.send(
                json.dumps(
                    {
                        "type": "ready",
                        "format": SUPPORTED_FORMAT,
                        "sample_rate": session.sample_rate,
                        "channels": 1,
                        "block_frame": session.block_frame,
                        "block_time": session.block_frame / session.sample_rate,
                        "algorithm_delay_ms": session.algorithm_delay_ms,
                    }
                )
            )

            async for message in websocket:
                if isinstance(message, str):
                    control = json.loads(message)
                    if control.get("type") == "flush":
                        continue
                    if control.get("type") == "close":
                        break
                    raise ValueError(f"unsupported control message: {control}")

                output_chunks = await asyncio.to_thread(
                    session.push_pcm,
                    message,
                    self.inference_lock,
                )
                for chunk in output_chunks:
                    await websocket.send(chunk)
        except websockets.ConnectionClosed:
            return
        except Exception as exc:
            error_payload = json.dumps({"type": "error", "message": str(exc)})
            try:
                await websocket.send(error_payload)
            finally:
                await websocket.close(code=1011)
            print(f"connection failed: {exc}")
        finally:
            if session is not None:
                print("session closed")

    def create_session(self, config: dict) -> RealtimePcmSession:
        fmt = config.get("format", SUPPORTED_FORMAT)
        if fmt != SUPPORTED_FORMAT:
            raise ValueError(f"unsupported format: {fmt}")
        reference_audio_path = config.get("reference_audio_path") or self.args.reference_audio_path
        if not reference_audio_path:
            raise ValueError("reference_audio_path is required")
        if not pathlib.Path(reference_audio_path).exists():
            raise ValueError(f"reference_audio_path does not exist: {reference_audio_path}")

        return RealtimePcmSession(
            rt_module=self.rt_module,
            model_set=self.model_set,
            device=self.device,
            reference_audio_path=reference_audio_path,
            sample_rate=int(config.get("sample_rate", 16000)),
            channels=int(config.get("channels", 1)),
            block_time=float(config.get("block_time", self.args.block_time)),
            crossfade_time=float(config.get("crossfade_time", self.args.crossfade_time)),
            extra_time_ce=float(config.get("extra_time_ce", self.args.extra_time_ce)),
            extra_time=float(config.get("extra_time", self.args.extra_time)),
            extra_time_right=float(config.get("extra_time_right", self.args.extra_time_right)),
            diffusion_steps=int(config.get("diffusion_steps", self.args.diffusion_steps)),
            inference_cfg_rate=float(config.get("inference_cfg_rate", self.args.inference_cfg_rate)),
            max_prompt_length=float(config.get("max_prompt_length", self.args.max_prompt_length)),
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Seed-VC tiny websocket PCM streaming server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--reference-audio-path", default=None)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--config-path", default="configs/presets/config_dit_mel_seed_uvit_xlsr_tiny.yml")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no-fp16", action="store_false", dest="fp16")
    parser.add_argument("--diffusion-steps", type=int, default=6)
    parser.add_argument("--inference-cfg-rate", type=float, default=0.7)
    parser.add_argument("--max-prompt-length", type=float, default=3.0)
    parser.add_argument("--block-time", type=float, default=0.18)
    parser.add_argument("--crossfade-time", type=float, default=0.04)
    parser.add_argument("--extra-time-ce", type=float, default=2.5)
    parser.add_argument("--extra-time", type=float, default=0.5)
    parser.add_argument("--extra-time-right", type=float, default=0.02)
    return parser.parse_args()


async def main_async():
    args = parse_args()
    server = SeedVcWebSocketServer(args)
    async with websockets.serve(
        server.handler,
        args.host,
        args.port,
        max_size=None,
        ping_interval=None,
    ):
        print(f"Seed-VC websocket PCM server listening on ws://{args.host}:{args.port}")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main_async())
