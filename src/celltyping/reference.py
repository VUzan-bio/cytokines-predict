from __future__ import annotations

from typing import Dict, List


def get_default_marker_defs() -> Dict[str, Dict[str, List[str]]]:
    """Return canonical PBMC marker definitions."""
    return {
        "CD4 T": {
            "positive": ["CD3D", "CD3E", "TRAC", "CD4", "IL7R"],
            "negative": ["CD8A", "NKG7", "MS4A1", "LYZ"],
        },
        "CD8 T": {
            "positive": ["CD3D", "CD3E", "TRAC", "CD8A", "CD8B"],
            "negative": ["CD4", "MS4A1", "LYZ"],
        },
        "NK": {
            "positive": ["NKG7", "GNLY", "PRF1", "TYROBP"],
            "negative": ["MS4A1", "LYZ"],
        },
        "B": {
            "positive": ["MS4A1", "CD79A", "CD79B", "CD37", "CD19"],
            "negative": ["LYZ", "NKG7"],
        },
        "Mono (classical)": {
            "positive": ["LYZ", "S100A8", "S100A9", "CTSS", "LGALS3", "FCGR3A"],
            "negative": ["MS4A1", "NKG7", "GNLY"],
        },
        "Mono (non-classical)": {
            "positive": ["LYZ", "FCGR3A", "LST1", "IFI30", "HLA-DPA1", "HLA-DPB1"],
            "negative": ["MS4A1", "NKG7", "GNLY"],
        },
        "DC": {
            "positive": ["FCER1A", "CST3", "LILRA4", "GZMB"],
            "negative": ["MS4A1", "NKG7"],
        },
        "Platelet": {
            "positive": ["PPBP", "PF4", "SDPR"],
            "negative": [],
        },
    }


def list_cell_types() -> List[str]:
    """List supported default cell types."""
    return list(get_default_marker_defs().keys())
