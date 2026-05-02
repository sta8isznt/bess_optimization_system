"""HeNEx DAM source file preprocessing."""

from __future__ import annotations

import glob
import os
from pathlib import Path

import pandas as pd


def henexing(folder_path: str | Path) -> pd.DataFrame:
    file_pattern = os.path.join(str(folder_path), "*.xlsx")
    all_files = glob.glob(file_pattern)

    if not all_files:
        print(f"Not found {folder_path}!")
        return pd.DataFrame(columns=["DELIVERY_MTU", "MCP"])

    frames = []
    for file in all_files:
        try:
            daily_df = pd.read_excel(file)
            frames.append(daily_df[["DELIVERY_MTU", "MCP"]].drop_duplicates())
        except Exception as exc:
            print(f"Error at {file}: {exc}")

    if not frames:
        return pd.DataFrame(columns=["DELIVERY_MTU", "MCP"])
    return pd.concat(frames, ignore_index=True)
