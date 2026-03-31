"""
File Utilities — Path helpers and temp file cleanup.
"""

import shutil
import tempfile
from pathlib import Path
from typing import List, Optional


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it doesn't exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_temp_dir(prefix: str = "vdub_") -> Path:
    """Create a temporary directory."""
    return Path(tempfile.mkdtemp(prefix=prefix))


def cleanup_dir(path: str | Path, keep_dir: bool = False):
    """Remove directory contents, optionally keeping the directory itself."""
    path = Path(path)
    if path.exists():
        if keep_dir:
            for item in path.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
        else:
            shutil.rmtree(path)


def safe_filename(name: str) -> str:
    """Convert string to a safe filename."""
    import re
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    name = name.strip('. ')
    return name or "unnamed"


def list_files(
    directory: str | Path,
    extensions: Optional[List[str]] = None,
    recursive: bool = False,
) -> List[Path]:
    """
    List files in a directory.

    Args:
        directory: Directory to list.
        extensions: Filter by extensions (e.g., ['.mp4', '.avi']).
        recursive: Whether to search recursively.

    Returns:
        Sorted list of file paths.
    """
    directory = Path(directory)
    if not directory.exists():
        return []

    if recursive:
        files = directory.rglob("*")
    else:
        files = directory.iterdir()

    result = [
        f for f in files
        if f.is_file()
        and (extensions is None or f.suffix.lower() in extensions)
    ]
    return sorted(result)


def get_file_size_mb(path: str | Path) -> float:
    """Get file size in megabytes."""
    return Path(path).stat().st_size / (1024 * 1024)
