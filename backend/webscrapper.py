"""Download face images for age-prediction datasets.

The script creates age-named folders like:
    ai/dataset/age_prediction/train/60/
    ai/dataset/age_prediction/train/61/
    ...
and fills each folder with a target number of images.

Example:
    python backend/webscrapper.py --start-age 60 --end-age 100 --images-per-age 1000

If you want a quick test before downloading the full dataset:
    python backend/webscrapper.py --start-age 60 --end-age 60 --images-per-age 3 --test-per-age 1 --dry-run
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Iterable, List, Set
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "age_images"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


def build_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=None,
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def clean_filename(raw_name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_name).strip("_")
    cleaned = cleaned or "image"
    if not cleaned.lower().endswith(VALID_EXTENSIONS):
        return f"{cleaned}.jpg"
    return cleaned


def search_queries_for_age(age: int) -> List[str]:
    base = [
        f"{age} years old person portrait",
        f"{age} year old man portrait",
        f"{age} year old woman portrait",
        f"middle age man {age} face",
        f"middle age woman {age} face",
        f"older person {age} face close up",
        f"senior man {age} portrait",
        f"senior woman {age} portrait",
        f"{age} old man photo",
        f"{age} old woman photo",
    ]
    return base


def extract_image_urls(html: str) -> List[str]:
    matches = []

    patterns = [
        r'"murl":"(https?://[^"\\]+)"',
        r'"imgurl":"(https?://[^"\\]+)"',
        r'"thumbnailUrl":"(https?://[^"\\]+)"',
        r'(https?://[^\s"\'<>]+\.(?:jpg|jpeg|png|webp)(?:\?[^\s"\'<>]*)?)',
    ]

    for pattern in patterns:
        for match in re.findall(pattern, html, flags=re.IGNORECASE):
            url = match.replace('\\u0026', '&').replace('\\/', '/')
            if url.startswith("http"):
                matches.append(url)

    unique: List[str] = []
    seen: Set[str] = set()
    for url in matches:
        lower = url.lower()
        if lower.endswith(VALID_EXTENSIONS):
            parsed = urlparse(url)
            if parsed.netloc and "bing" not in parsed.netloc:
                if url not in seen:
                    seen.add(url)
                    unique.append(url)
    return unique


def download_image(url: str, output_path: Path, session: requests.Session) -> bool:
    try:
        response = session.get(url, headers=HEADERS, timeout=25, stream=True)
        if response.status_code != 200:
            return False

        content_type = response.headers.get("Content-Type", "").lower()
        if "image" not in content_type and not output_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            return False

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return True
    except Exception:
        return False


def collect_age_images(age: int, dest_dir: Path, target_count: int, session: requests.Session, delay: float = 0.6) -> int:
    dest_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    seen_links: Set[str] = set()

    for query in search_queries_for_age(age):
        if downloaded >= target_count:
            break

        try:
            search_url = "https://www.bing.com/images/search"
            response = session.get(search_url, params={"q": query}, headers=HEADERS, timeout=20)
            if response.status_code != 200:
                continue

            image_urls = extract_image_urls(response.text)
            if not image_urls:
                continue

            for image_url in image_urls:
                if downloaded >= target_count:
                    break
                if image_url in seen_links:
                    continue
                seen_links.add(image_url)

                file_name = f"{age}_{downloaded + 1:04d}_{int(time.time())}.jpg"
                target_path = dest_dir / clean_filename(file_name)
                if download_image(image_url, target_path, session):
                    downloaded += 1
                    if delay > 0:
                        time.sleep(delay)
                else:
                    if target_path.exists():
                        target_path.unlink(missing_ok=True)

        except Exception:
            continue

        if delay > 0:
            time.sleep(delay * 0.5)

    return downloaded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download age-specific face image folders for age prediction training.")
    parser.add_argument("--start-age", type=int, default=60, help="Starting age. Default: 60")
    parser.add_argument("--end-age", type=int, default=100, help="Ending age. Default: 100")
    parser.add_argument("--images-per-age", type=int, default=100, help="How many images to download per age folder. Default: 100")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Folder to save the downloaded images in. Default: backend/age_images")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without downloading anything")
    parser.add_argument("--delay", type=float, default=0.8, help="Seconds to wait between image downloads. Default: 0.8")
    return parser.parse_args()


def ensure_dataset_dirs(output_dir: Path, start_age: int, end_age: int) -> None:
    for age in range(start_age, end_age + 1):
        (output_dir / str(age)).mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()

    if args.start_age > args.end_age:
        print("Error: --start-age cannot be greater than --end-age.", file=sys.stderr)
        return 2

    if args.images_per_age <= 0:
        print("Error: --images-per-age must be greater than 0.", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir).resolve()
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output folder: {output_dir}")
    print(f"Age range: {args.start_age} to {args.end_age}")
    print(f"Images per age: {args.images_per_age}")

    if args.dry_run:
        total_images = (args.end_age - args.start_age + 1) * args.images_per_age
        print("Dry run: no images downloaded.")
        print(f"Expected images: {total_images}")
        print(f"Folders to create under: {output_dir}")
        return 0

    session = build_session()
    ensure_dataset_dirs(output_dir, args.start_age, args.end_age)

    total_downloaded = 0
    for age in range(args.start_age, args.end_age + 1):
        age_dir = output_dir / str(age)
        train_count = collect_age_images(age, age_dir, args.images_per_age, session, delay=args.delay)
        total_downloaded += train_count
        print(f"Age {age}: downloaded {train_count} images in {age_dir}.")

    print("\nFinished.")
    print(f"Total downloaded images: {total_downloaded}")
    print(f"Files saved in: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
