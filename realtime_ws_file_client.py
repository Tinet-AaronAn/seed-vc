import argparse
import asyncio
import json
import math
import time

import librosa
import numpy as np
import soundfile as sf
import websockets


def float32_to_pcm_s16le(wav: np.ndarray) -> bytes:
    wav = np.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)
    wav = np.clip(wav, -1.0, 1.0)
    return (wav * 32767.0).astype("<i2").tobytes()


def pcm_s16le_to_float32(data: bytes) -> np.ndarray:
    if len(data) % 2 != 0:
        data = data[:-1]
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


async def run_client(args):
    global_started_at = time.perf_counter()
    audio, _ = librosa.load(args.input_wav, sr=args.sample_rate, mono=True)
    frame_samples = max(1, int(args.frame_ms * args.sample_rate / 1000))
    expected_block_samples = max(1, int(args.block_time * args.sample_rate / (args.sample_rate // 50)) * (args.sample_rate // 50))
    expected_chunks = audio.size // expected_block_samples
    output_chunks = []
    first_output_at = None
    all_output_event = asyncio.Event()

    def stamp(label):
        now = time.perf_counter()
        print(f"[{label}] +{now - global_started_at:.3f}s")
        return now

    config = {
        "format": "pcm_s16le",
        "sample_rate": args.sample_rate,
        "channels": 1,
        "reference_audio_path": args.reference_audio_path,
        "block_time": args.block_time,
        "crossfade_time": args.crossfade_time,
        "extra_time_ce": args.extra_time_ce,
        "extra_time": args.extra_time,
        "extra_time_right": args.extra_time_right,
        "diffusion_steps": args.diffusion_steps,
        "inference_cfg_rate": args.inference_cfg_rate,
        "max_prompt_length": args.max_prompt_length,
    }

    stamp("client_start")
    print(
        f"input: samples={audio.size} duration={audio.size / args.sample_rate:.3f}s "
        f"frame_samples={frame_samples} expected_block_samples={expected_block_samples} "
        f"expected_chunks={expected_chunks}"
    )

    stamp("connect_start")
    async with websockets.connect(args.url, max_size=None, ping_interval=None) as websocket:
        stamp("connect_done")
        await websocket.send(json.dumps(config))
        ready = json.loads(await websocket.recv())
        if ready.get("type") != "ready":
            raise RuntimeError(f"server did not become ready: {ready}")
        stamp("server_ready")
        print(f"server ready: {ready}")

        async def receiver():
            nonlocal first_output_at
            async for message in websocket:
                if isinstance(message, str):
                    event = json.loads(message)
                    if event.get("type") == "error":
                        raise RuntimeError(event["message"])
                    print(f"server event: {event}")
                    continue
                if first_output_at is None:
                    first_output_at = stamp("first_output_chunk")
                output_chunks.append(pcm_s16le_to_float32(message))
                if len(output_chunks) >= expected_chunks:
                    all_output_event.set()

        receiver_task = asyncio.create_task(receiver())
        stamp("send_start")
        for offset in range(0, audio.size, frame_samples):
            frame = audio[offset : offset + frame_samples]
            await websocket.send(float32_to_pcm_s16le(frame))
            if args.realtime:
                await asyncio.sleep(frame.size / args.sample_rate)

        stamp("send_done")
        try:
            await asyncio.wait_for(all_output_event.wait(), timeout=args.drain_seconds)
            stamp("all_expected_output_received")
        except asyncio.TimeoutError:
            stamp("drain_timeout")
        print(f"received_chunks={len(output_chunks)} expected_chunks={expected_chunks}")
        receiver_task.cancel()
        await asyncio.gather(receiver_task, return_exceptions=True)
        try:
            await websocket.send(json.dumps({"type": "close"}))
            await asyncio.wait_for(websocket.close(), timeout=2.0)
            stamp("websocket_closed")
        except Exception as exc:
            stamp("websocket_close_skipped")
            print(f"close warning: {exc}")

    if output_chunks:
        output = np.concatenate(output_chunks)
    else:
        output = np.empty(0, dtype=np.float32)
    sf.write(args.output_wav, output, args.sample_rate)
    stamp("wav_written")
    elapsed = time.perf_counter() - global_started_at
    print(
        f"wrote {args.output_wav}: chunks={len(output_chunks)}/{expected_chunks} "
        f"duration={output.size / args.sample_rate:.2f}s elapsed={elapsed:.2f}s"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Seed-VC websocket PCM file streaming client")
    parser.add_argument("--url", default="ws://127.0.0.1:8765")
    parser.add_argument("--input-wav", required=True)
    parser.add_argument("--reference-audio-path", required=True)
    parser.add_argument("--output-wav", default="output_ws.wav")
    parser.add_argument("--sample-rate", type=int, choices=[8000, 16000], default=16000)
    parser.add_argument("--frame-ms", type=int, default=20)
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--drain-seconds", type=float, default=2.0)
    parser.add_argument("--diffusion-steps", type=int, default=6)
    parser.add_argument("--inference-cfg-rate", type=float, default=0.7)
    parser.add_argument("--max-prompt-length", type=float, default=3.0)
    parser.add_argument("--block-time", type=float, default=0.18)
    parser.add_argument("--crossfade-time", type=float, default=0.04)
    parser.add_argument("--extra-time-ce", type=float, default=2.5)
    parser.add_argument("--extra-time", type=float, default=0.5)
    parser.add_argument("--extra-time-right", type=float, default=0.02)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run_client(parse_args()))
