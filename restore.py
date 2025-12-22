import os, json, shutil
from pathlib import Path

TARGET = Path(".").expanduser().resolve()

HISTORY_ROOTS = [
    Path("~/.vscode-server/data/User/History").expanduser(),
    Path("~/.vscode-server-insiders/data/User/History").expanduser(),
    Path("~/.vscode-remote/data/User/History").expanduser(),
]

def file_uri_to_path(s: str) -> Path:
    # handles file:/// URIs minimally
    if s.startswith("file://"):
        s = s[len("file://"):]
    s = s.replace("%20", " ")
    return Path(s)

def newest_content_from_folder(folder: Path):
    """
    Heuristic for VS Code Local History:
    - folder contains entries.json plus one or more content blobs
    - entries.json often includes original resource path/URI and a list of entries with ids/timestamps
    """
    entries = folder / "entries.json"
    if not entries.exists():
        return None

    try:
        data = json.loads(entries.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None

    resource = None
    ent_list = None

    if isinstance(data, dict):
        resource = data.get("resource") or data.get("source") or data.get("path")
        ent_list = data.get("entries") or data.get("items")
    elif isinstance(data, list):
        ent_list = data
    else:
        return None

    if not isinstance(resource, str) or not resource:
        return None
    orig_path = file_uri_to_path(resource)
    try:
        orig_path = orig_path.expanduser().resolve()
    except Exception:
        return None

    if not ent_list:
        return None

    # pick newest entry by timestamp-ish fields
    def ts(e):
        if not isinstance(e, dict):
            return 0
        return int(e.get("timestamp") or e.get("time") or e.get("modifiedTime") or 0)

    best = max((e for e in ent_list if isinstance(e, dict)), key=ts, default=None)
    if best is None:
        return None

    cid = best.get("id") or best.get("name") or best.get("entry") or best.get("location")
    content_path = None

    # Attempt to resolve content file path
    if isinstance(cid, str):
        p = folder / cid
        if p.exists():
            content_path = p
        else:
            # try common variants
            for suf in ["", ".txt", ".json", ".bin"]:
                p2 = folder / (cid + suf)
                if p2.exists():
                    content_path = p2
                    break

    # fallback: newest file (excluding entries.json)
    if content_path is None or not content_path.exists():
        files = [p for p in folder.iterdir() if p.is_file() and p.name != "entries.json"]
        if not files:
            return None
        content_path = max(files, key=lambda p: p.stat().st_mtime)

    return orig_path, content_path

def restore():
    if not TARGET.exists():
        print(f"TARGET does not exist: {TARGET}\nCreate it (empty is fine) and re-run.")
        return

    restored = 0
    considered = 0

    for root in HISTORY_ROOTS:
        if not root.exists():
            continue
        for folder in root.iterdir():
            if not folder.is_dir():
                continue

            res = newest_content_from_folder(folder)
            if res is None:
                continue
            orig_path, content_path = res

            # Only restore files that belong under TARGET
            if TARGET != orig_path and TARGET not in orig_path.parents:
                continue

            considered += 1
            orig_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copyfile(content_path, orig_path)
                restored += 1
            except Exception:
                pass

    print(f"Done. Considered {considered} history items under TARGET.")
    print(f"Restored {restored} files into {TARGET}.")
    print("Next: inspect results, then rebuild git metadata (clone/init).")

if __name__ == "__main__":
    restore()

