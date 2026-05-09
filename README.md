# SkinMapper neural reconstruction — RunPod deployment

Self-hosted 2DGS + SAM 2 + Depth Anything v2 + MASt3R pipeline running on
RunPod serverless GPUs.

```
photos.zip
  → SAM 2 (limb-only mask)
  → MASt3R (camera poses)
  → Depth Anything v2 (depth priors)
  → 2DGS training (10k iterations, ~5-10 min on RTX 4090)
  → mesh extraction + dual-texture baking
result.zip   {mesh.obj, mesh.mtl, mesh_original.png, mesh_albedo.png}
```

## Files in this folder

| File | Role |
|---|---|
| `Dockerfile` | Container build |
| `requirements.txt` | Python deps |
| `handler.py` | RunPod serverless entry point |
| `pipeline/segment.py` | SAM 2 + GroundingDINO segmentation |
| `pipeline/pose.py` | MASt3R pose estimation |
| `pipeline/depth.py` | Depth Anything v2 priors |
| `pipeline/train.py` | 2DGS training |
| `pipeline/extract.py` | Mesh + texture baking + delighting |
| `pipeline/utils.py` | HEIC, blur detection, helpers |

The Mac-side client lives one level up: `../runpod_client.py`.

---

## One-time deployment

You'll do this in three stages: build container → deploy serverless endpoint
→ wire endpoint ID into `.env`.

### Stage 1 — Build the container image

The container is ~12-15 GB once built. Two options:

**Option A — RunPod's image builder (recommended).** Push this folder to a
private GitHub repo, then point RunPod at it.

1. Create a private GitHub repo, e.g. `skinmapper-pipeline`
2. From your Mac:
   ```bash
   cd /Users/oleishaproksa/Desktop/skinmapper-mac-test/runpod_pipeline
   git init
   git add .
   git commit -m "initial pipeline"
   git remote add origin git@github.com:goobersoober/skinmapper-pipeline.git
   git push -u origin main
   ```
3. Go to **RunPod → Settings → Container Registry Auth** and add a GitHub
   personal access token (read access to the repo) — it's used as a
   Container Registry credential.
4. Continue to Stage 2; you'll point the endpoint at the Dockerfile in this
   repo.

**Option B — Docker Hub.** Build locally on a Linux machine with NVIDIA
hardware, push to Docker Hub. Skip if you don't have one. Aaron's Mac can't
build a CUDA image, so use Option A.

### Stage 2 — Create the serverless endpoint

1. Go to https://runpod.io/console/serverless and click **New Endpoint**
2. Settings:
   - **Name**: `skinmapper-recon`
   - **Source**: GitHub → pick the repo from Stage 1
   - **Branch**: `main`
   - **Dockerfile path**: `Dockerfile`
   - **Container Disk**: `30 GB` (the model checkpoints alone are ~6 GB)
   - **GPU**: **RTX 4090** (or A40 / A6000 as fallback). Avoid A100 — 2× cost
     for marginal speed win on this workload.
   - **Workers**:
     - `Min workers`: `0`  (only spin up on demand → no idle cost)
     - `Max workers`: `2`  (lift later if you get concurrent submissions)
     - `Idle timeout`: `30 s`
     - `Execution timeout`: `1800 s` (30 min — well over our worst case)
   - **Container start command**: leave blank (Dockerfile CMD is correct)
   - **Environment variables**: leave blank for now
3. Click **Deploy**. RunPod will pull the repo, build the image, and stand
   up workers. **First build takes 25-40 min** because of the heavy ML deps.
   Subsequent rebuilds use layer caching and take 3-5 min.
4. Once the endpoint shows status "Ready", **copy the Endpoint ID** — it's
   the random string in the endpoint URL like `xxxxxxxxxxxxxx`.

### Stage 3 — Wire it into the Mac server

Edit `/Users/oleishaproksa/Desktop/skinmapper-mac-test/.env`:

```
RUNPOD_API_KEY=your_api_key_here
RUNPOD_ENDPOINT_ID=the_id_you_just_copied
```

Restart `mac_server.py`. The new `/submit` flow now hits RunPod first and
falls back to Apple Object Capture if RunPod fails.

---

## First scan checklist

After deployment, run a single test scan to confirm the wiring:

```bash
cd /Users/oleishaproksa/Desktop/skinmapper-mac-test
/opt/homebrew/bin/python3.11 -c "
from pathlib import Path
from runpod_client import RunPodReconstructor
import shutil, zipfile, tempfile

# Build a small photo zip from test_scans/scan_average
src = Path('test_scans/scan_average')
tmp = Path(tempfile.mkdtemp())
zp  = tmp / 'photos.zip'
with zipfile.ZipFile(zp, 'w', zipfile.ZIP_STORED) as z:
    for p in src.iterdir():
        z.write(p, arcname=p.name)
print(f'zip: {zp.stat().st_size/1e6:.1f} MB')

rc = RunPodReconstructor()
out = rc.run(zp, scan_type='design', body_part='leg')
print('OK →', out)
"
```

**Watch for:**
- `[runpod] submitting … MB zip` — confirms credentials work
- `[runpod] <id> IN_QUEUE` → `IN_PROGRESS` → `COMPLETED`
- A `result.zip` next to your photos zip
- Inspect `result.zip` — should contain `mesh.obj`, `mesh.mtl`,
  `mesh_albedo.png` (or `mesh_original.png`)

---

## Cost & latency check after first scan

Open the endpoint in RunPod's web UI → click on the executed job. You'll see:
- Execution time (target: 12-18 min)
- GPU type used
- Cost (target: $0.15-0.25)

If execution time is > 25 min or cost is > $0.50, drop the iteration count
in `runpod_client.run(..., iterations=8000)` or downgrade GPU to A40.

---

## Limitations & known iteration points

This is a v1 pipeline. Expected first-deployment fixes:

1. **CUDA / PyTorch version drift.** The 2DGS submodule (`diff-surfel-rasterization`)
   sometimes needs a specific PyTorch version. If the build fails at
   `pip install submodules/diff-surfel-rasterization`, pin PyTorch to whatever
   the upstream repo's CI uses.
2. **MASt3R pose convention.** The world-vs-camera convention conversion in
   `pose.py` is best-effort — if poses come out mirrored or upside down,
   flip the inversion in `_quat_to_rot` callers.
3. **GroundingDINO prompt tuning.** If SAM 2 segments the wrong region,
   adjust `PROMPT_BY_BODY_PART` in `segment.py`. Usually adding more
   discriminative noun phrases fixes it.
4. **Texture seams.** First passes often have visible seams where photos
   overlap. The `cylindrical_remap.py` step on the Mac will replace the UV
   layout entirely, so atlas seams from this stage don't reach the user.
5. **Inline base64 size limit.** RunPod's REST API caps input at ~10 MB.
   For larger photo sets, switch to presigned URLs (R2 / S3).

---

## Updating the pipeline

After you push code changes:

```bash
git add . && git commit -m "..." && git push
```

In RunPod web UI: endpoint → "Refresh from GitHub" → wait for rebuild.
Existing in-flight jobs finish on the old image; new jobs get the new one.
