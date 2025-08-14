# HK3 Parquet Lab — UI Guide

A lightweight web UI for exploring parquet datasets and visualizing narrative graphs.

## Quick start

- Requirements: Python 3.12+, uv package manager
- Install deps (local venv recommended):

```
# Windows (PowerShell)
uv venv .venv --clear
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt

# macOS/Linux
uv venv .venv --clear
source .venv/bin/activate
uv pip install -r requirements.txt
```

- Run the server:
```
# Windows (PowerShell)
$env:UV_SYSTEM_PYTHON=1
uv run --with fastapi==0.111.0 --with "uvicorn[standard]==0.30.1" --with polars==1.5.0 \
  uvicorn server:app --host 127.0.0.1 --port 8000 --reload

# macOS/Linux
UV_SYSTEM_PYTHON=1 \
uv run --with fastapi==0.111.0 --with "uvicorn[standard]==0.30.1" --with polars==1.5.0 \
  uvicorn server:app --host 127.0.0.1 --port 8000 --reload
```

Open `http://127.0.0.1:8000`.

## UI overview

- Schema panel
  - Paste or edit a Pydantic-style schema; Save Schema stores it server-side for Modal calls.
  - Templates: insert predefined Action / NarrativeAnalysis examples.

- Upload panel
  - Drag-and-drop a `.parquet` file or click to select.
  - A small preview table is rendered (first rows only).
  - Each column has an ellipsis button `...` to expand content for that entire column.

- Console panel
  - Shows action logs (schema save, Modal calls, build feedback, errors).

- Narrative Graph panel
  - Scan Modal Volume: lists `.parquet` files found in `modal_volume_processed/` (or `modal_processed/`).
  - File chips: click to select a Modal file for graph building.
  - Controls:
    - Build Graph: all source↔target interactions from the first N rows (N = Rows input).
    - Characters Only: character↔character edges only (uses `source_is_character` / `target_is_character` flags when present).
    - Clean View: same as Characters Only + simple clean-up (split composite names; drop pronouns; keep title-case names).
    - Story Mode / Stop: progressively builds the graph row-by-row up to N, showing the story’s progression.
    - Rows: number of rows to include (10–500).
    - Rotate / Zoom chips: toggle Plotly 3D drag modes (orbit/turntable).
  - Graph display:
    - 3D Plotly scatter3d; node labels are fixed at 14px.
    - Panel height is 800px (adjustable in `static/index.html` via `#graphPanel`).
    - The camera view persists across reloads (saved to `localStorage`). Pan/zoom as you like; it will be restored automatically.

## Graph modes in detail

- All Interactions (Build Graph)
  - Nodes: unique names found in `source` and `target`.
  - Edges: undirected; multiple occurrences aggregated as weights.

- Characters Only
  - Filters to rows where both `source_is_character` and `target_is_character` are true (if such columns exist).

- Clean View
  - Applies simple split on composite names (e.g., "A and B" → [A, B]).
  - Removes pronouns / generic terms and keeps title-cased names.

- Story Mode
  - Rebuilds the first 1..N rows incrementally with a small delay so the graph “grows like a story”. Use Stop to cancel.

## Tips

- Drag controls: use the Rotate/Zoom chips or Plotly’s modebar to adjust the camera.
- Reset camera: clear the browser’s `localStorage` for this site to remove saved camera settings.
- Table preview: click the `...` button to expand a column; overflow indicators auto-hide when text fits.

## Troubleshooting

- 400 errors fetching a graph slice usually mean: missing filename, invalid mode, or the slice returned zero rows (window too small/large).
- If the graph appears off-center, the fixed camera and aspect mode are set to keep it centered; adjust the view once and it will persist.
- If Plotly labels are too small/large, change sizes in `static/main.js` (`textfont.size` and `layout.font.size`).

## Project notes

- Backend: FastAPI, endpoints mounted at `/api/*`.
- Frontend: vanilla JS + Plotly 3D.
- Data: read with Polars; we detect columns automatically (`source`, `target`, `source_is_character`, `target_is_character`, and an embedding list column when present).
- Embeddings mode exists server-side; the slim UI focuses on interactions/character views and a simple story mode for clarity and speed. 

[note]
The actions buttons need to be linked up with the data...

The Generate Extract Schema and Embeddings button needs to be linked up to backend modal...
we use embeddings already generated, but it's a case of hooking up the old scripts

Darg and Drop creates local versions of the files