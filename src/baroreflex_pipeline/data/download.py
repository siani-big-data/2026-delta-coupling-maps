from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse
from urllib.request import Request, urlopen

from .. import __version__

_USER_AGENT = f"baroreflex-representation-pipeline/{__version__}"


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self.links.append(href)


def _directory_links(url: str) -> list[str]:
    body: str | None = None
    for attempt in range(6):
        try:
            request = Request(url, headers={"User-Agent": _USER_AGENT})
            with urlopen(request, timeout=45) as response:
                encoding = response.headers.get_content_charset() or "utf-8"
                body = response.read().decode(encoding, errors="replace")
            break
        except Exception as error:
            if attempt == 5:
                raise
            delay = min(30, 2**attempt)
            print(
                f"[RETRY] PhysioNet index unavailable ({type(error).__name__}); "
                f"retrying in {delay}s",
                flush=True,
            )
            time.sleep(delay)
    assert body is not None
    parser = _LinkParser()
    parser.feed(body)
    return parser.links


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    for attempt in range(5):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            headers["User-Agent"] = _USER_AGENT
            request = Request(url, headers=headers)
            with urlopen(request, timeout=60) as response:
                resumed = offset > 0 and getattr(response, "status", None) == 206
                content_length = response.headers.get("Content-Length")
                expected_transfer_size = int(content_length) if content_length else None
                if resumed:
                    content_range = response.headers.get("Content-Range", "")
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+|\*)", content_range)
                    if match is None or int(match.group(1)) != offset:
                        raise IOError(
                            f"Server returned an invalid resume range for {destination.name}: "
                            f"{content_range!r}, expected offset {offset}"
                        )
                mode = "ab" if resumed else "wb"
                with partial.open(mode) as output:
                    shutil.copyfileobj(response, output, length=1024 * 1024)
            if expected_transfer_size is not None:
                expected_partial_size = (offset if resumed else 0) + expected_transfer_size
                if partial.stat().st_size != expected_partial_size:
                    raise IOError(
                        f"Incomplete transfer for {destination.name}: "
                        f"{partial.stat().st_size} != {expected_partial_size} bytes"
                    )
            expected_wfdb_size = _expected_wfdb_size(destination)
            if expected_wfdb_size is not None and partial.stat().st_size != expected_wfdb_size:
                raise IOError(
                    f"WFDB size mismatch for {destination.name}: "
                    f"{partial.stat().st_size} != {expected_wfdb_size} bytes"
                )
            partial.replace(destination)
            return
        except Exception as error:
            if attempt == 4:
                raise
            time.sleep(2**attempt)
            partial_size = partial.stat().st_size if partial.exists() else 0
            print(
                f"[RETRY] {destination.name}: attempt {attempt + 2}/5 after "
                f"{type(error).__name__}; saved={partial_size / 1024**2:.1f} MiB",
                flush=True,
            )


def _expected_wfdb_size(path: Path) -> int | None:
    if path.suffix != ".dat":
        return None
    header = path.with_suffix(".hea")
    if not header.exists():
        return None
    header_lines = header.read_text(encoding="utf-8", errors="replace").splitlines()
    fields = header_lines[0].split()
    if len(fields) < 4:
        return None
    signal_count = int(fields[1])
    sample_count = int(fields[3])
    signal_lines = header_lines[1 : 1 + signal_count]
    formats = [line.split()[1].split("x", 1)[0].split(":", 1)[0] for line in signal_lines]
    if formats and all(item == "16" for item in formats):
        return signal_count * sample_count * 2
    return None


def _is_complete_file(path: Path) -> bool:
    if not path.exists():
        return False
    expected = _expected_wfdb_size(path)
    return expected is None or path.stat().st_size == expected


def _stored_bytes(files: list[tuple[str, Path, str]]) -> int:
    total = 0
    for _, output, _ in files:
        if output.exists():
            total += output.stat().st_size
            continue
        partial = output.with_name(output.name + ".part")
        if partial.exists():
            total += partial.stat().st_size
    return total


def _progress_message(
    indexed_files: list[tuple[str, str]],
    destination: Path,
    pending_count: int,
    started_at: float,
    initial_bytes: int,
) -> str:
    complete_files = sum(_is_complete_file(destination / relative) for _, relative in indexed_files)
    dat_files = [relative for _, relative in indexed_files if relative.endswith(".dat")]
    complete_dat = sum(_is_complete_file(destination / relative) for relative in dat_files)
    tracked = [
        (candidate, destination / relative, relative) for candidate, relative in indexed_files
    ]
    current_bytes = _stored_bytes(tracked)
    elapsed = max(time.monotonic() - started_at, 0.001)
    rate = max(current_bytes - initial_bytes, 0) / elapsed
    rate_mib = rate / 1024**2
    details = (
        f"files={complete_files}/{len(indexed_files)}, "
        f"records={complete_dat}/{len(dat_files)}, "
        f"stored={current_bytes / 1024**2:.1f} MiB, "
        f"speed={rate_mib:.2f} MiB/s, pending={pending_count}"
    )
    if len(dat_files) == 1121 and rate > 0:
        expected_bytes = 5.9 * 1024**3
        fraction = min(current_bytes / expected_bytes, 1.0)
        remaining_seconds = max(expected_bytes - current_bytes, 0) / rate
        details += f", approx={fraction:.1%}, ETA~{remaining_seconds / 3600:.1f}h"
    return f"[DOWNLOAD PROGRESS] {details}"


def _manifest_files(root_url: str, manifest: Path) -> list[tuple[str, str]]:
    with manifest.open(newline="", encoding="utf-8") as source:
        record_ids = [str(row["record_id"]).zfill(4) for row in csv.DictReader(source)]
    relative_files = ["subject-info.csv"]
    relative_files.extend(
        f"{record_id}.{extension}" for record_id in record_ids for extension in ("hea", "dat")
    )
    return [(urljoin(root_url, relative), relative) for relative in relative_files]


def _crawl_files(root_url: str) -> list[tuple[str, str]]:
    root = urlparse(root_url)
    pending = [root_url]
    visited: set[str] = set()
    indexed_files: list[tuple[str, str]] = []
    while pending:
        directory_url = pending.pop()
        if directory_url in visited:
            continue
        visited.add(directory_url)
        for href in _directory_links(directory_url):
            candidate = urljoin(directory_url, href)
            parsed = urlparse(candidate)
            if parsed.scheme not in {"http", "https"} or parsed.netloc != root.netloc:
                continue
            if not parsed.path.startswith(root.path) or parsed.query or parsed.fragment:
                continue
            relative = unquote(parsed.path[len(root.path) :]).lstrip("/")
            if not relative or any(part in {"", ".", ".."} for part in Path(relative).parts):
                continue
            if candidate.endswith("/"):
                pending.append(candidate)
            else:
                indexed_files.append((candidate, relative))
    return indexed_files


def _exclude_missing_files(
    indexed_files: list[tuple[str, str]], manifest: Path
) -> list[tuple[str, str]]:
    with manifest.open(newline="", encoding="utf-8") as source:
        known_missing = {
            f"{str(row['record_id']).zfill(4)}.dat"
            for row in csv.DictReader(source)
            if "absent from release" in row.get("reason", "").lower()
        }
    if known_missing:
        print(
            f"[DOWNLOAD] Skipping {len(known_missing)} file(s) documented as absent "
            f"from the release: {', '.join(sorted(known_missing))}",
            flush=True,
        )
    return [(url, relative) for url, relative in indexed_files if relative not in known_missing]


def _indexed_files(root_url: str, destination: Path) -> list[tuple[str, str]]:
    index_path = destination / ".download-index.json"
    packaged_manifest = Path(__file__).resolve().parents[2] / "cohort" / "manifest.csv"
    if index_path.exists():
        cached = json.loads(index_path.read_text(encoding="utf-8"))
        indexed_files = [(item["url"], item["relative"]) for item in cached["files"]]
    elif "autonomic-aging-cardiovascular/1.0.0" in root_url and packaged_manifest.exists():
        indexed_files = _manifest_files(root_url, packaged_manifest)
    else:
        indexed_files = _crawl_files(root_url)
    if "autonomic-aging-cardiovascular/1.0.0" in root_url and packaged_manifest.exists():
        indexed_files = _exclude_missing_files(indexed_files, packaged_manifest)
    index_path.write_text(
        json.dumps(
            {
                "source": root_url,
                "files": [{"url": item, "relative": relative} for item, relative in indexed_files],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return indexed_files


def _pending_files(
    indexed_files: list[tuple[str, str]], destination: Path
) -> list[tuple[str, Path, str]]:
    files: list[tuple[str, Path, str]] = []
    for candidate, relative in indexed_files:
        output = destination / relative
        if _is_complete_file(output):
            continue
        if output.exists():
            expected = _expected_wfdb_size(output)
            print(
                f"[INVALID] {relative}: local_size={output.stat().st_size}, "
                f"expected_size={expected}; downloading a clean replacement.",
                flush=True,
            )
        files.append((candidate, output, relative))
    return files


def _download_with_python(url: str, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root_url = url if url.endswith("/") else url + "/"
    indexed_files = _indexed_files(root_url, destination)

    files = _pending_files(indexed_files, destination)

    if not files:
        print("[DOWNLOAD] Dataset is already complete.", flush=True)
        return
    worker_count = min(6, len(files))
    initial_bytes = _stored_bytes(
        [(candidate, destination / relative, relative) for candidate, relative in indexed_files]
    )
    started_at = time.monotonic()
    print(
        f"[DOWNLOAD] Resuming with {worker_count} workers; "
        f"{len(indexed_files) - len(files)}/{len(indexed_files)} files already complete; "
        f"{initial_bytes / 1024**2:.1f} MiB stored.",
        flush=True,
    )

    remaining = files
    for round_number in range(1, 4):
        failures: list[tuple[str, Path, str]] = []
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(_download_file, candidate, output): (candidate, output, relative)
                for candidate, output, relative in remaining
            }
            pending = set(futures)
            last_progress_report = started_at
            while pending:
                completed, pending = wait(pending, timeout=15, return_when=FIRST_COMPLETED)
                if not completed:
                    print(
                        _progress_message(
                            indexed_files, destination, len(pending), started_at, initial_bytes
                        ),
                        flush=True,
                    )
                    last_progress_report = time.monotonic()
                    continue
                for future in completed:
                    item = futures[future]
                    try:
                        future.result()
                    except Exception as error:
                        failures.append(item)
                        print(
                            f"[DEFERRED] {item[2]} after repeated {type(error).__name__}; "
                            "other files will continue.",
                            flush=True,
                        )
                    else:
                        print(f"[DOWNLOADED] {item[2]}", flush=True)
                if time.monotonic() - last_progress_report >= 15:
                    print(
                        _progress_message(
                            indexed_files,
                            destination,
                            len(pending) + len(failures),
                            started_at,
                            initial_bytes,
                        ),
                        flush=True,
                    )
                    last_progress_report = time.monotonic()
        if not failures:
            return
        remaining = failures
        print(
            f"[DOWNLOAD] Retry round {round_number + 1}/3 for {len(remaining)} deferred files.",
            flush=True,
        )
    names = ", ".join(relative for _, _, relative in remaining[:10])
    raise RuntimeError(
        f"PhysioNet remained unavailable for {len(remaining)} files after three rounds. "
        f"Partial data were preserved. First files: {names}"
    )


def download_dataset(url: str, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    executable = shutil.which("wget")
    if executable is None:
        _download_with_python(url, destination)
        return

    command = [
        executable,
        "--recursive",
        "--no-parent",
        "--continue",
        "--timestamping",
        "--no-host-directories",
        "--cut-dirs=3",
        "--reject=index.html*",
        f"--directory-prefix={destination}",
        url,
    ]
    subprocess.run(command, check=True)
