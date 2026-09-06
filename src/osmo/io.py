"""Immutable raw source retrieval and reproducibility records."""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def session():
    s = requests.Session()
    s.headers['User-Agent'] = 'OSMO-research/0.1 (source audit and PK research)'
    s.mount('https://', HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504])))
    return s


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(tmp, p)


def download(url, path, expected_sha256=None):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        temp = p.with_suffix(p.suffix + '.part')
        with session().get(url, stream=True, timeout=(20, 120)) as r:
            r.raise_for_status()
            with temp.open('wb') as f:
                for chunk in r.iter_content(1024 * 1024):
                    f.write(chunk)
        os.replace(temp, p)
    sha = digest(p)
    if expected_sha256 and sha != expected_sha256:
        raise ValueError(f'Source hash mismatch: {p.name}')
    return dict(url=url, sha256=sha, bytes=p.stat().st_size,
                accessed_utc=datetime.now(timezone.utc).isoformat())
