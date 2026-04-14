"""
collection/channel_loader.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Read channel targets from an Excel file.

Expected Excel columns (case-insensitive):
  - channel_id     : YouTube channel ID (e.g. UCxxxxxx)     [optional if handle given]
  - channel_handle : YouTube handle    (e.g. @channelname)  [optional if id given]
  - name           : Human-readable label                   [optional]

At least one of channel_id or channel_handle must be non-empty per row.
"""

import logging
from pathlib import Path
from typing import TypedDict

import pandas as pd

logger = logging.getLogger(__name__)


class ChannelEntry(TypedDict):
    channel_id: str
    channel_handle: str
    name: str


def load_channels_from_excel(path: str) -> list[ChannelEntry]:
    """
    Parse *path* (.xlsx / .xls) and return a list of ChannelEntry dicts.
    Raises FileNotFoundError if the file does not exist.
    """
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(
            f"Channels file not found: {path}\n"
            "Run 'python create_channels_template.py' to create a sample file."
        )

    df = pd.read_excel(file, dtype=str)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    if "channel_id" not in df.columns and "channel_handle" not in df.columns:
        raise ValueError(
            f"Excel file must contain at least a 'channel_id' or 'channel_handle' column. "
            f"Found columns: {list(df.columns)}"
        )

    for col in ("channel_id", "channel_handle", "name"):
        if col not in df.columns:
            df[col] = ""

    df = df.fillna("")

    entries: list[ChannelEntry] = []
    for i, row in df.iterrows():
        ch_id = str(row["channel_id"]).strip()
        ch_handle = str(row["channel_handle"]).strip()
        ch_name = str(row.get("name", "")).strip()

        if not ch_id and not ch_handle:
            logger.warning("Row %d skipped — both channel_id and channel_handle are empty.", i + 2)
            continue

        entries.append(
            ChannelEntry(channel_id=ch_id, channel_handle=ch_handle, name=ch_name)
        )

    logger.info("Loaded %d channel(s) from %s", len(entries), path)
    return entries
