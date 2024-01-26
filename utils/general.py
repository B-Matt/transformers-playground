import os
import torch
import urllib
import glob

from pathlib import Path
from collections import namedtuple

# Constants
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # Root directory
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))

# Vars
data_info_tuple = namedtuple(
    'data_info_tuple',
    'name, image, mask'
)

# Functions
def check_suffix(file='', suffix=('.pt',), msg=''):
    """
    Check given files for acceptable suffix.
    """
    if file and suffix:
        if isinstance(suffix, str):
            suffix = [suffix]

        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower()  # file suffix
            if len(s):
                assert s in suffix, f"{msg}{f} acceptable suffix is {suffix}"


def check_file(file, suffix=''):
    """
    Search or download file (if is necessary) and returns files path.
    """
    if file.startswith(('rtsp:/')):
        return f'rtspsrc location={str(file)} latency=500 ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink'

    check_suffix(file, suffix)
    file = str(file)

    if os.path.isfile(file) or not file:            # Already exists
        return file

    elif file.startswith(('http:/', 'https:/')):    # Start download
        url = file
        file = Path(urllib.parse.unquote(file).split('?')[0]).name  # '%2F' to '/', split https://url.com/file.txt?auth
        if os.path.isfile(file):
            return file
        
        torch.hub.download_url_to_file(url, file)
        assert Path(file).exists() and Path(file).stat().st_size > 0, f'File download failed: {url}'  # Check did we downloaded file?
        return file

    else:
        files = []
        for d in 'data', 'models', 'utils':
            files.extend(glob.glob(str(ROOT / d / '**' / file), recursive=True))  # Find file

            assert len(files), f'File not found: {file}'  # Assert file wasn't found
            assert len(files) == 1, f"Multiple files match '{file}', specify exact path: {files}"  # Assert unique file
            return files[0]  # Return file
