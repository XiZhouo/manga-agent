# Windows LaMa Deployment Guide

This project does not need to fully migrate to Windows yet.

Recommended split:

- Keep the orchestrator / Brain / RAG workflow on your current macOS machine.
- Run only the LaMa HTTP service on the Windows machine with the `3090`.
- Let macOS call Windows over LAN via `MANGA_AGENT_LAMA_URL`.

That gives you the GPU where it matters, without forcing the whole agent stack to move first.

## Recommended rollout order

1. Keep developing the agent on macOS.
2. Copy the repo to Windows when you are ready to validate real inpainting.
3. Start only `skills/lama_erase_server/server.py` on Windows.
4. Point the macOS orchestrator to that server.
5. Run smoke tests, then run real pages such as `test3`.

## What to copy to Windows

Simplest option: copy the whole repo.

Example layout on Windows:

```text
D:\agent1\
  orchestrator\
  skills\
  scripts\
  README.md
```

You can also copy a narrower subset later, but for now the full repo is easier and avoids path drift.

## Windows setup

### 1) Create a venv

```powershell
cd D:\agent1
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

### 2) Install project dependencies

```powershell
pip install -r requirements.txt
pip install fastapi uvicorn pillow opencv-python numpy requests
```

If you use `simple_lama`, also install its package in this env.

## Two supported LaMa backends

### Option A: `simple_lama`

Use this first if you want the easiest path.

Set:

```powershell
$env:LAMA_RUNNER="simple_lama"
```

### Option B: official `saicinpainting_cli`

Use this when you want the closer-to-official LaMa path.

Required env vars:

```powershell
$env:LAMA_RUNNER="saicinpainting_cli"
$env:LAMA_REPO_DIR="D:\models\lama"
$env:LAMA_MODEL_DIR="D:\models\big-lama"
$env:LAMA_DEVICE="cuda"
```

Optional:

```powershell
$env:LAMA_PYTHON_EXE="D:\agent1\.venv\Scripts\python.exe"
```

## Start the Windows LaMa server

From the repo root on Windows:

```powershell
$env:LAMA_RUNNER="saicinpainting_cli"
$env:LAMA_REPO_DIR="D:\models\lama"
$env:LAMA_MODEL_DIR="D:\models\big-lama"
$env:LAMA_DEVICE="cuda"
python -m uvicorn skills.lama_erase_server.server:app --host 0.0.0.0 --port 8765
```

If you are testing with `simple_lama`, swap only `LAMA_RUNNER`.

## Point macOS orchestrator to Windows

On macOS, set:

```bash
export MANGA_AGENT_LAMA_URL=http://<windows-lan-ip>:8765
export MANGA_AGENT_LAMA_BACKEND=lama
```

Example:

```bash
export MANGA_AGENT_LAMA_URL=http://192.168.1.88:8765
export MANGA_AGENT_LAMA_BACKEND=lama
```

## Smoke checks

### On Windows

Open:

- `http://127.0.0.1:8765/health`
- `http://127.0.0.1:8765/capabilities`

You want to see:

- `ok: true`
- `runner: simple_lama` or `runner: saicinpainting_cli`

### From macOS

Replace the IP below with your Windows LAN IP:

```bash
curl http://192.168.1.88:8765/health
curl http://192.168.1.88:8765/capabilities
```

## How this fits your current project

The current erase path is:

1. VLM returns `bbox_xywh` + `has_text_box`
2. macOS builds the page mask locally
3. macOS saves debug artifacts in `masks/`
4. macOS sends `image + final mask + profile` to Windows LaMa
5. Windows returns the erased page
6. macOS caches `erased/page_XXXX.png`
7. lettering reuses that erased page for retries or translation edits

So the migration boundary is already clean: only the inpainting service has to move.

## Debugging `test3` specifically

For pages where text is embedded in artwork:

- check `masks/page_XXXX_overlay.png`
- check `masks/page_XXXX_refined.png`
- if glyph edges are missing, increase:
  - `mask_dilate_px`
  - `embedded_box_expand_px`
  - `embedded_complete_mask_close_px`
- if nearby art is being damaged, decrease:
  - `mask_dilate_px`
  - `embedded_box_expand_px`
  - `embedded_complete_mask_close_px`
  - or switch `profile` toward `conservative`

## Practical advice

Do not migrate the full agent runtime to Windows yet unless you truly need to.

Right now the best next step is:

- keep coding on macOS
- deploy only LaMa on Windows
- validate real erase quality on your hard cases
- only then decide whether the whole orchestrator should move
