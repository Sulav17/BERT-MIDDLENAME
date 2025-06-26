"""
Build label mappings dynamically from JSON data files.
Loads all JSON files in the project 'data' directory and extracts unique "middle_name" values.
"""
import json
from pathlib import Path

# Locate the data directory (project_root/data)
_project_root = Path(__file__).resolve().parents[2]
_data_dir = _project_root / "data"

# Collect unique middle names from all JSON files in data directory
_names = set()
for _json_path in _data_dir.glob("*.json"):
    try:
        with open(_json_path, encoding="utf-8") as _f:
            _entries = json.load(_f)
    except Exception:
        continue
    if isinstance(_entries, list):
        for _entry in _entries:
            _m = _entry.get("middle_name")
            if _m:
                _names.add(_m)

# Sort for deterministic ordering
middle_names = sorted(_names)
print(middle_names)

# Build mappings
middle_name_to_id = {name: idx for idx, name in enumerate(middle_names)}
id_to_middle_name = {idx: name for name, idx in middle_name_to_id.items()}
