# EBQwen Basketball Court Grounding

This module turns E-BARD's object grounding into **court-line semantic grounding**.
The VLM emits pixel-space lines/polylines; it does not use bounding boxes for long
or curved markings.

## Output contract

```json
{
  "court_lines": [
    {
      "label": "baseline",
      "type": "line",
      "points": [[88, 701], [1187, 492]]
    },
    {
      "label": "three_point_line",
      "type": "polyline",
      "points": [[251, 613], [324, 566], [408, 536]]
    }
  ]
}
```

Allowed labels are `sideline`, `baseline`, `half_court_line`,
`free_throw_line`, `paint_boundary`, `three_point_line`, and `center_circle`.
Coordinates are pixels in the original image, with `[0, 0]` at top-left.

## 1. Prepare a zero-shot dataset

Copy `data/test_court_grounding.example.json` to
`data/test_court_grounding.json`, put images under `data/images/`, and add one
sample per image. Image paths are resolved relative to the JSON file. The empty
`gpt` value is allowed during zero-shot inference.

Validate paths before reserving a GPU:

```bash
python -m court_grounding.src.validate_dataset \
  --data court_grounding/data/test_court_grounding.json \
  --allow-empty-ground-truth
```

## 2. Run inference on a GPU server

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r court_grounding/requirements.txt

python -m court_grounding.src.infer \
  --model /models/EBQwen2.5-VL-3B \
  --processor Qwen/Qwen2.5-VL-3B-Instruct \
  --data court_grounding/data/test_court_grounding.json \
  --output court_grounding/outputs/predictions.json
```

If the fine-tuned model includes processor files, omit `--processor`. Use
`--limit 5` for a smoke test. Malformed model responses are retained in
`prediction_raw` and explained in `parse_error`, so one bad sample does not stop
the batch.

## 3. Visualize results

```bash
python -m court_grounding.src.visualize \
  --predictions court_grounding/outputs/predictions.json \
  --output-dir court_grounding/outputs/visualized
```

Review semantic label, coordinate accuracy, JSON validity, curves, and occlusion.
Only after this check should annotated SFT data be produced. For SFT, fill each
`gpt` value with the exact output contract above and validate without
`--allow-empty-ground-truth`.

## 4. Recommended training progression

1. Zero-shot test on 20-50 varied broadcast frames.
2. Annotate 200-500 frames with visible line endpoints/polyline samples.
3. Mix court-grounding records with the existing E-BARD SFT tasks to reduce
   catastrophic forgetting.
4. Fine-tune the existing EBQwen checkpoint with LoRA/QLoRA in the Qwen2.5-VL
   SFT repository referenced by the root README.
5. Evaluate on games/arenas not present in training, then consider adding court
   keypoints and homography for metric court coordinates.

Do not commit model weights, raw videos, private datasets, or generated output.

## 5. Git and server update

Push code from the development machine (replace `YOUR_REPOSITORY_URL` with a
repository where you have write access):

```bash
git remote rename origin upstream
git remote add origin YOUR_REPOSITORY_URL
git push -u origin codex/court-grounding
```

After merging that branch to `main`, deploy on the GPU server:

```bash
ssh USER@SERVER
cd /path/to/E-BARD
git status --short
git pull --ff-only origin main
source .venv/bin/activate
pip install -r court_grounding/requirements.txt
python -m court_grounding.src.validate_dataset \
  --data court_grounding/data/test_court_grounding.json \
  --allow-empty-ground-truth
python -m court_grounding.src.infer \
  --model /models/EBQwen2.5-VL-3B \
  --processor Qwen/Qwen2.5-VL-3B-Instruct \
  --data court_grounding/data/test_court_grounding.json \
  --output court_grounding/outputs/predictions.json \
  --limit 5
```

`git pull --ff-only` deliberately refuses to overwrite divergent server-side
changes. Keep models and datasets outside Git; copy them separately or mount
shared storage. Remove `--limit 5` after the smoke test succeeds.

## 6. DeepSeek Flash API baseline

This adapter first asks which markings are unambiguously visible, then makes one
localization request per visible label. It automatically saves both JSON and a
visualized image.

```bash
pip install -r court_grounding/requirements.txt
export DEEPSEEK_API_KEY="replace-with-your-key"

python -m court_grounding.src.infer_deepseek \
  --data court_grounding/data/test_court_grounding.json \
  --output court_grounding/outputs/predictions_deepseek.json \
  --visualize-dir court_grounding/outputs/visualized_deepseek \
  --limit 1
```

The visualization is written to
`court_grounding/outputs/visualized_deepseek/00000_example.jpg`. Do not put the
API key in source code, dataset JSON, shell scripts, or Git. Each test image is
sent to DeepSeek. Raw presence and per-label localization responses are retained
in the output JSON for auditing.

For a server-local Python key file, create the Git-ignored file
`court_grounding/src/deepseek_key_local.py` containing:

```python
DEEPSEEK_API_KEY = "replace-with-your-key"
```

Do not put the key in `infer_deepseek.py`; that tracked file is updated by Git,
so local edits to it block future pulls.

For a single arbitrary image, no dataset JSON is needed:

```bash
python -m court_grounding.src.infer_deepseek \
  --image /absolute/path/to/frame.jpg \
  --visualize-dir court_grounding/outputs/visualized_deepseek
```

### Windows one-image test

From PowerShell at the repository root:

```powershell
.\court_grounding\run_deepseek_windows.ps1 `
  -ImagePath "C:\path\to\frame.jpg" `
  -Detail high
```

The script checks the local key file, runs inference, and writes both JSON and
the rendered image under `court_grounding\outputs\windows\<image-name>\`.
