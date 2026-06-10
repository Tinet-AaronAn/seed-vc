# Realtime WebSocket PCM API

This server wraps the existing `real-time-gui.py` tiny-model realtime path and exposes
streaming PCM over WebSocket.

## Start server

```bash
python realtime_ws_server.py \
  --host 0.0.0.0 \
  --port 8765 \
  --checkpoint-path DiT_uvit_tat_xlsr_ema.pth \
  --config-path configs/presets/config_dit_mel_seed_uvit_xlsr_tiny.yml \
  --reference-audio-path examples/reference/dingzhen_0.wav \
  --diffusion-steps 6 \
  --block-time 0.18
```

If `--checkpoint-path` is omitted, the server uses the same Hugging Face default
loading path as `real-time-gui.py`.

## Protocol

The first WebSocket message must be a JSON text config:

```json
{
  "format": "pcm_s16le",
  "sample_rate": 16000,
  "channels": 1,
  "reference_audio_path": "examples/reference/dingzhen_0.wav",
  "block_time": 0.18,
  "crossfade_time": 0.04,
  "extra_time_ce": 2.5,
  "extra_time": 0.5,
  "extra_time_right": 0.02,
  "diffusion_steps": 6,
  "inference_cfg_rate": 0.7,
  "max_prompt_length": 3.0
}
```

Supported input and output audio format:

- `format`: `pcm_s16le`
- `sample_rate`: `8000` or `16000`
- `channels`: mono is preferred; multi-channel input is mixed down to mono

The server replies with a JSON `ready` event:

```json
{
  "type": "ready",
  "format": "pcm_s16le",
  "sample_rate": 16000,
  "channels": 1,
  "block_frame": 2880,
  "block_time": 0.18,
  "algorithm_delay_ms": 380
}
```

After that, clients send binary `pcm_s16le` frames. The server returns binary
`pcm_s16le` frames at the same sample rate.

## File streaming test

```bash
python realtime_ws_file_client.py \
  --url ws://127.0.0.1:8765 \
  --input-wav examples/source/source_s1.wav \
  --reference-audio-path examples/reference/dingzhen_0.wav \
  --output-wav output_ws_16k.wav \
  --sample-rate 16000 \
  --realtime
```

For 8 kHz:

```bash
python realtime_ws_file_client.py \
  --url ws://127.0.0.1:8765 \
  --input-wav examples/source/source_s1.wav \
  --reference-audio-path examples/reference/dingzhen_0.wav \
  --output-wav output_ws_8k.wav \
  --sample-rate 8000 \
  --realtime
```

## Realtime requirement

For stable realtime output, watch the server log:

```text
processed block: sr=16000 frames=2880 diffusion_steps=6 elapsed_ms=...
```

The practical requirement is:

```text
p95 elapsed_ms < block_time * 1000 * 0.8
```

If inference is slower than the block duration, increase `block_time`, lower
`diffusion_steps`, or move to a faster GPU.
