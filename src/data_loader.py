"""
MobiAct dataset loader and segmenter.

Handles the confirmed dataset structure:
- 20 activity/fall/scenario folders under Annotated Data/
- Filenames: <CODE>_<SUBJECT_ID>_<TRIAL_NO>_annotated.csv
- Each file may contain multiple labels in sequence (e.g. fall files
  are STD -> FALL_CODE -> LYI); segmentation must respect these
  in-file label transitions, never window across a boundary.
"""

import os
import re
import pandas as pd

FILENAME_PATTERN = re.compile(r'^([A-Za-z]+)_(\d+)_(\d+)_annotated\.csv$', re.IGNORECASE)

FALL_CODES = ['BSC', 'FKL', 'FOL', 'SDL']
ADL_CODES_9 = ['STD', 'WAL', 'JOG', 'JUM', 'STU', 'STN', 'SCH', 'CSI', 'CSO']
ADL_CODES_11 = ADL_CODES_9 + ['SIT', 'CHU']
SCENARIO_CODES = ['SBE', 'SBW', 'SLH', 'SLW', 'SRH']  # excluded from training (confirmed: field standard)
EXCLUDED_CODES = SCENARIO_CODES  # LYI is fall-derived, handled separately if needed


class MobiActLoader:
    """
    Parameters
    ----------
    data_root : str
        Path to '.../MobiAct_Dataset_v2.0/Annotated Data'
    """

    def __init__(self, data_root: str):
        self.data_root = data_root
        if not os.path.exists(data_root):
            raise FileNotFoundError(
                f"MobiAct annotated data not found at {data_root}. "
                f"Extract the dataset first."
            )

    def load_file(self, code: str, subject: int, trial: int = 1) -> pd.DataFrame | None:
        """Load one raw annotated CSV. Returns None if missing."""
        path = os.path.join(self.data_root, code, f"{code}_{subject}_{trial}_annotated.csv")
        if not os.path.exists(path):
            return None
        return pd.read_csv(path)

    def get_segment(self, code: str, subject: int, trial: int = 1) -> pd.DataFrame | None:
        """
        Load a file and return only the rows matching the requested
        activity/fall code's own label — i.e. strip any lead-in (STD)
        or trailing (LYI) segments present in fall/transition files.
        """
        df = self.load_file(code, subject, trial)
        if df is None:
            return None
        seg = df[df['label'] == code].reset_index(drop=True)
        return seg if len(seg) > 0 else None

    def list_subjects(self, code: str) -> list[int]:
        """List subject IDs available for a given activity code."""
        folder = os.path.join(self.data_root, code)
        if not os.path.exists(folder):
            return []
        subjects = set()
        for fname in os.listdir(folder):
            m = FILENAME_PATTERN.match(fname)
            if m:
                subjects.add(int(m.group(2)))
        return sorted(subjects)

    def get_label_transitions(self, code: str, subject: int, trial: int = 1) -> list[str]:
        """
        Return the collapsed sequence of labels present in a raw file
        (e.g. ['STD', 'BSC', 'LYI']). Useful for validating the
        segmentation logic against the confirmed dataset structure.
        """
        df = self.load_file(code, subject, trial)
        if df is None:
            return []
        labels = df['label'].tolist()
        collapsed = [labels[0]]
        for lbl in labels[1:]:
            if lbl != collapsed[-1]:
                collapsed.append(lbl)
        return collapsed
