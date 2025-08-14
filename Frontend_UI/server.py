from __future__ import annotations

import os
import pathlib
import uuid
from typing import List, Dict, Any, Optional
import math

import polars as pl
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import asyncio
import sys


ROOT_DIR = pathlib.Path(__file__).parent.resolve()
STATIC_DIR = ROOT_DIR / "static"
UPLOADS_DIR = ROOT_DIR / "uploads"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def _safe_filename(original_name: str) -> str:
    base = pathlib.Path(original_name).name.replace("\x00", "")
    # Prevent hidden paths and ensure parquet extension
    name, ext = os.path.splitext(base)
    if ext.lower() != ".parquet":
        raise HTTPException(status_code=400, detail="Only .parquet files are supported")
    unique = uuid.uuid4().hex[:8]
    return f"{name}.{unique}.parquet"


def _df_preview(df: pl.DataFrame, max_rows: int = 100) -> Dict[str, Any]:
    preview_df = df.head(max_rows)
    columns: List[str] = list(preview_df.columns)
    rows: List[Dict[str, Any]] = [
        {col: (val.item() if isinstance(val, pl.Series) else val) for col, val in zip(columns, row)}
        for row in preview_df.rows()
    ]
    return {
        "columns": columns,
        "first_rows": rows,
        "num_cols": len(columns),
        "num_rows": len(rows),
    }


app = FastAPI(title="HK3 Parquet Lab")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.post("/api/upload-parquet")
async def upload_parquet(file: UploadFile = File(...)) -> JSONResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    try:
        safe_name = _safe_filename(file.filename)
    except HTTPException as e:
        raise e
    save_path = UPLOADS_DIR / safe_name

    # Persist file to disk
    with save_path.open("wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    # Build preview using Polars (limited rows)
    try:
        df = pl.read_parquet(str(save_path), n_rows=1000)
    except Exception as exc:
        # Clean up bad file
        try:
            save_path.unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:
            pass
        raise HTTPException(status_code=400, detail=f"Failed to read parquet: {exc}")

    preview = _df_preview(df, max_rows=100)

    return JSONResponse({
        "ok": True,
        "filename": safe_name,
        **preview,
    })


# Modal processed volume support
def _get_modal_base_dir() -> pathlib.Path:
    """Resolve the local directory that represents the Modal processed volume.

    Preference order:
    1) modal_volume_processed
    2) modal_processed
    Falls back to creating modal_volume_processed if neither exists.
    """
    preferred = ROOT_DIR / "modal_volume_processed"
    legacy = ROOT_DIR / "modal_processed"
    if preferred.exists() and preferred.is_dir():
        return preferred
    if legacy.exists() and legacy.is_dir():
        return legacy
    preferred.mkdir(parents=True, exist_ok=True)
    return preferred


@app.get("/api/modal-files")
def list_modal_files() -> JSONResponse:
    base_dir = _get_modal_base_dir()
    files = []
    for p in sorted(base_dir.glob("*.parquet")):
        try:
            stat = p.stat()
            files.append({
                "name": p.name,
                "size": stat.st_size,
            })
        except OSError:
            # Skip files that cannot be accessed
            continue
    return JSONResponse({"ok": True, "files": files})


@app.get("/api/modal-file-info")
def modal_file_info(file: Optional[str] = None) -> JSONResponse:
    base_dir = _get_modal_base_dir()
    if not file:
        return JSONResponse({"ok": False, "error": "Missing 'file' query param"}, status_code=400)
    safe_name = pathlib.Path(file).name
    target = base_dir / safe_name
    if not (target.exists() and target.is_file() and target.suffix.lower() == ".parquet"):
        return JSONResponse({"ok": False, "error": "File not found or invalid"}, status_code=400)
    try:
        lf = pl.scan_parquet(str(target))
        count_df = lf.select(pl.len().alias("n")).collect()
        total_rows = int(count_df["n"][0])
    except Exception as exc:
        return JSONResponse({"ok": False, "error": f"Failed to read parquet: {exc}"}, status_code=400)
    return JSONResponse({"ok": True, "rows": total_rows})


@app.get("/api/embeddings-graph")
def embeddings_graph(
    file: Optional[str] = None,
    offset: int = 0,
    length: int = 200,
    top_k: int = 3,
    threshold: float = 0.35,
    mode: str = "auto",
    clean: bool = False,
) -> JSONResponse:
    """Return a lightweight similarity graph or a sequential fallback derived from a parquet file slice.

    Args:
        file: Parquet filename from modal volume directory.
        offset: Starting row index.
        length: Number of rows to include (capped to 500).
        top_k: Top neighbors per node to keep (embeddings mode).
        threshold: Minimum cosine similarity to include an edge (embeddings mode).
        mode: One of {"auto", "embeddings", "sequential", "interactions", "interactions_all"}.
            - auto: prefer embeddings if available; else interactions; else sequential
            - embeddings: force embeddings if available; else fallback to sequential
            - sequential: force sequential row-to-row edges
            - interactions: force source↔target character interactions when columns exist; else fallback to sequential
            - interactions_all: force all source↔target interactions (ignores character flags)
        clean: When True with interactions modes, split composite names and drop pronouns/non-titlecase
    """
    # Input validation
    assert offset >= 0, "offset must be non-negative"
    assert length > 0, "length must be positive"
    assert 1 <= top_k <= 50, "top_k must be between 1 and 50"
    assert 0.0 <= threshold <= 1.0, "threshold must be in [0,1]"
    mode = (mode or "auto").strip().lower()
    if mode not in {"auto", "embeddings", "sequential", "interactions", "interactions_all"}:
        return JSONResponse({"ok": False, "error": f"Invalid mode: {mode}"}, status_code=400)

    base_dir = _get_modal_base_dir()
    if not file:
        return JSONResponse({"ok": False, "error": "Missing 'file' query param"}, status_code=400)

    safe_name = pathlib.Path(file).name
    target = base_dir / safe_name
    if not (target.exists() and target.is_file() and target.suffix.lower() == ".parquet"):
        return JSONResponse({"ok": False, "error": "File not found or invalid"}, status_code=400)

    try:
        lf = pl.scan_parquet(str(target))
    except Exception as exc:
        return JSONResponse({"ok": False, "error": f"Failed to read parquet: {exc}"}, status_code=400)

    schema = lf.schema
    if not schema:
        return JSONResponse({"ok": False, "error": "Parquet has no columns"}, status_code=400)

    # Helpers
    def _is_float_list_dtype(dt: pl.DataType) -> bool:
        return dt == pl.List(pl.Float32) or dt == pl.List(pl.Float64)

    def _is_string_dtype(dt: pl.DataType) -> bool:
        dt_str = str(dt)
        return dt_str in ("Utf8", "String")

    # Simple splitter and beautifier
    STOP_CHARACTERS = {
        "he","she","it","they","someone","anyone","everybody","anybody","person","thing","creature","man","woman","boy","girl","people","nobody","nothing","narrator","herself","himself","hers","him","themselves","us","we","you","yourself","yourselves","i","me","myself","itself"
    }

    def _split_name(name: str) -> List[str]:
        s = name.replace("\u2013","-")
        for sep in [" and ", " & ", ",", ";", "/", " + ", " with ", " vs ", " versus "]:
            s = s.replace(sep, "|")
        parts = [p.strip() for p in s.split("|")]
        out: List[str] = []
        seen: set[str] = set()
        for p in parts:
            if not p:
                continue
            if p in seen:
                continue
            seen.add(p)
            out.append(p)
        return out

    def _beautify(names: List[str]) -> List[str]:
        cleaned: List[str] = []
        for n in names:
            low = n.strip().lower()
            if not n or low in STOP_CHARACTERS:
                continue
            # require title-case-like token (first letter uppercase)
            if not n[0].isupper():
                continue
            cleaned.append(n)
        return cleaned

    # Column detection
    embed_col_name: Optional[str] = None
    for c, dt in schema.items():
        if _is_float_list_dtype(dt):
            embed_col_name = c
            break

    preferred_labels = [
        "text", "chunk_text", "content", "sentence", "action", "title",
        "doc_id", "id",
    ]

    label_col_name: Optional[str] = None
    for name in preferred_labels:
        if name in schema and _is_string_dtype(schema[name]):
            label_col_name = name
            break
    if label_col_name is None:
        for c, dt in schema.items():
            if _is_string_dtype(dt):
                label_col_name = c
                break

    # Read slice
    max_length = min(int(length), 500)
    start_offset = max(0, int(offset))
    try:
        sample_df = lf.slice(start_offset, max_length).collect()
    except Exception as exc:
        return JSONResponse({"ok": False, "error": f"Failed to slice parquet: {exc}"}, status_code=400)
    if sample_df.height == 0:
        return JSONResponse({"ok": False, "error": "Requested slice returned no rows"}, status_code=400)

    # Decide which mode to run
    treat_as_no_embeddings = (embed_col_name is None) or (mode in {"sequential", "interactions", "interactions_all"})

    # Fallback modes if no embeddings or forced mode: interactions by source/target if present, else sequential rows
    if treat_as_no_embeddings:
        has_source = "source" in sample_df.columns
        has_target = "target" in sample_df.columns
        if (mode in {"interactions", "interactions_all", "auto"}) and has_source and has_target:
            node_set: Dict[str, Dict[str, Any]] = {}
            edges_count: Dict[tuple[str, str], int] = {}
            has_src_char = "source_is_character" in sample_df.columns
            has_tgt_char = "target_is_character" in sample_df.columns
            for row in sample_df.iter_rows(named=True):
                if mode == "interactions":
                    if has_src_char and has_tgt_char:
                        if not bool(row.get("source_is_character")) or not bool(row.get("target_is_character")):
                            continue
                s_raw = row.get("source")
                t_raw = row.get("target")
                if s_raw is None or t_raw is None:
                    continue
                s_name = str(s_raw).strip()
                t_name = str(t_raw).strip()
                if not s_name or not t_name:
                    continue
                s_parts = [s_name]
                t_parts = [t_name]
                if clean:
                    s_parts = _split_name(s_name)
                    t_parts = _split_name(t_name)
                    s_parts = _beautify(s_parts)
                    t_parts = _beautify(t_parts)
                if not s_parts or not t_parts:
                    continue
                for sp in s_parts:
                    for tp in t_parts:
                        if sp == tp:
                            continue
                        if sp not in node_set:
                            node_set[sp] = {"id": sp, "label": sp, "group": 1}
                        if tp not in node_set:
                            node_set[tp] = {"id": tp, "label": tp, "group": 1}
                        a, b = (sp, tp) if sp < tp else (tp, sp)
                        edges_count[(a, b)] = edges_count.get((a, b), 0) + 1
            nodes: List[Dict[str, Any]] = list(node_set.values())
            edges: List[Dict[str, Any]] = [{"source": a, "target": b, "weight": float(c)} for (a, b), c in edges_count.items()]
            return JSONResponse({"ok": True, "start_offset": start_offset, "nodes": nodes, "edges": edges})

        # Sequential mode
        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []
        labels: List[str] = []
        for row in sample_df.iter_rows(named=True):
            if label_col_name is not None:
                val = row.get(label_col_name)
                labels.append(str(val) if val is not None else "")
            else:
                labels.append("")
        for i in range(sample_df.height):
            label = labels[i] if i < len(labels) else ""
            if label and len(label) > 120:
                label = label[:117] + "..."
            nodes.append({
                "id": f"r{start_offset + i}",
                "label": label or f"Row {i}",
                "group": 1,
            })
        for i in range(sample_df.height - 1):
            edges.append({
                "source": f"r{start_offset + i}",
                "target": f"r{start_offset + i + 1}",
                "weight": 0.2,
            })
        return JSONResponse({"ok": True, "start_offset": start_offset, "nodes": nodes, "edges": edges})

    # Embeddings mode
    vectors: List[List[float]] = []
    labels: List[str] = []
    for row in sample_df.iter_rows(named=True):
        vec = row.get(embed_col_name)  # type: ignore[arg-type]
        if not isinstance(vec, list) or not vec:
            continue
        vec_floats = [float(x) for x in vec]
        norm = math.sqrt(sum(x * x for x in vec_floats))
        if norm == 0.0:
            continue
        vec_unit = [x / max(norm, 1e-12) for x in vec_floats]
        vectors.append(vec_unit)
        if label_col_name is not None:
            val = row.get(label_col_name)
            labels.append(str(val) if val is not None else "")
        else:
            labels.append("")

    num_nodes = len(vectors)
    if num_nodes < 2:
        return JSONResponse({"ok": False, "error": "Not enough valid embedding rows"}, status_code=400)

    neighbors: List[List[tuple[int, float]]] = [[] for _ in range(num_nodes)]
    for i in range(num_nodes):
        vi = vectors[i]
        for j in range(i + 1, num_nodes):
            vj = vectors[j]
            sim = sum(a * b for a, b in zip(vi, vj))
            if sim >= threshold:
                neighbors[i].append((j, sim))
                neighbors[j].append((i, sim))

    edge_set = set()
    for i in range(num_nodes):
        nbrs = sorted(neighbors[i], key=lambda t: t[1], reverse=True)[:top_k]
        for j, _ in nbrs:
            a, b = (i, j) if i < j else (j, i)
            edge_set.add((a, b))

    nodes: List[Dict[str, Any]] = []
    for i in range(num_nodes):
        label = labels[i] if i < len(labels) else ""
        if label and len(label) > 120:
            label = label[:117] + "..."
        nodes.append({
            "id": f"r{start_offset + i}",
            "label": label or f"Row {i}",
            "group": 1,
        })

    edges: List[Dict[str, Any]] = []
    for a, b in edge_set:
        sim_val: Optional[float] = None
        for j, s in neighbors[a]:
            if j == b:
                sim_val = s
                break
        if sim_val is None:
            for j, s in neighbors[b]:
                if j == a:
                    sim_val = s
                    break
        weight = float(sim_val) if sim_val is not None else 0.3
        edges.append({
            "source": f"r{start_offset + a}",
            "target": f"r{start_offset + b}",
            "weight": max(0.05, weight),
        })

    return JSONResponse({"ok": True, "start_offset": start_offset, "nodes": nodes, "edges": edges})


@app.post("/api/schema")
async def save_schema(payload: Dict[str, Any]) -> JSONResponse:
    schema_text = str(payload.get("schema", ""))
    return JSONResponse({"ok": True, "length": len(schema_text)})


@app.post("/api/modal/extract-schema")
async def modal_extract_schema(payload: Dict[str, Any]) -> JSONResponse:
    """Spawn the parent-level book processor as a background process.

    The UI calls this to kick off extraction without blocking.
    Optional payload keys (ignored by script unless it parses argv):
      book_start, num_books, parallel_books, max_calls, max_batch_size
    """
    parent_dir = ROOT_DIR.parent
    script_path = parent_dir / "book_processor.py"
    if not script_path.exists():
        return JSONResponse({"ok": False, "error": f"Script not found: {script_path}"}, status_code=400)

    # Optional numeric args (best-effort); if not used by script's __main__, it will ignore
    args: List[str] = []
    def _add_arg(key: str, flag: str):
        val = payload.get(key)
        if val is None:
            return
        try:
            args.extend([f"--{flag}", str(int(val))])
        except Exception:
            pass

    _add_arg("book_start", "book_start")
    _add_arg("num_books", "num_books")
    _add_arg("parallel_books", "parallel_books")
    _add_arg("max_batch_size", "max_batch_size")
    if payload.get("max_calls") is not None:
        try:
            args.extend(["--max_calls", str(int(payload["max_calls"]))])
        except Exception:
            pass

    async def _spawn():
        try:
            await asyncio.create_subprocess_exec(
                sys.executable,
                str(script_path),
                *args,
                cwd=str(parent_dir),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                stdin=asyncio.subprocess.DEVNULL,
                creationflags=(0x00000008 if os.name == 'nt' else 0),  # DETACHED_PROCESS on Windows
            )
        except Exception:
            # Ignore spawn errors for now; client gets immediate response anyway
            pass

    asyncio.create_task(_spawn())
    return JSONResponse({"ok": True, "message": "Processor started"})
