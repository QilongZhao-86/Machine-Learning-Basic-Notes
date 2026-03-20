#!/usr/bin/env python3
"""
Download a mixed list of research URLs and paper titles.

What it does
- Direct URLs: saves the HTML page and tries to discover/download a linked PDF.
- Paper titles: searches OpenAlex first, then arXiv, then Crossref.
- Saves metadata to metadata.json and a summary CSV.

Usage
  python download_video_action_papers.py
  python download_video_action_papers.py --out downloads
  python download_video_action_papers.py --items my_items.txt

items.txt format
- One item per line.
- Can be either a URL or a paper title.
- Blank lines are ignored.

Dependencies
  pip install requests beautifulsoup4 feedparser
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urljoin, urlparse

import feedparser
import requests
from bs4 import BeautifulSoup

DEFAULT_ITEMS = [
    "https://www.rhoda.ai/research/direct-video-action",
    "S-VAM: Shortcut Video-Action Model by Self-Distilling Geometric and Semantic Foresight",
    "Fast-WAM: Do World Action Models Need Test-time Future Imagination?",
    "Video Prediction Policy: A Generalist Robot Policy with Predictive Visual Representations",
    "Video Generators are Robot Policies",
    "mimic-video: Video-Action Models for Generalizable Robot Control Beyond VLAs",
    "Lingbot-VA: Causal World Modeling for Robot Control",
    "COSMOS POLICY: FINE-TUNING VIDEO MODELS FOR VISUOMOTOR CONTROL AND PLANNING",
    "Unified Video Action Model",
    "GR-2: A Generative Video-Language-Action Model with Web-Scale Knowledge for Robot Manipulation",
    "GAIA-2: A Controllable Multi-View Generative World Model for Autonomous Driving",
    "Prediction with Action: Visual Policy Learning via Joint Denoising Process",
    "Motus: A Unified Latent Action World Model",
    "Unified World Models: Coupling Video and Action Diffusion for Pretraining on Large Robotic Datasets",
    "LATENT ACTION PRETRAINING FROM VIDEOS",
    "WorldVLA: Towards Autoregressive Action World Model",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

PDF_HEADERS = {
    **HEADERS,
    "Accept": "application/pdf,*/*;q=0.8",
}

TIMEOUT = 30
SLEEP_SECONDS = 1.0


@dataclass
class Result:
    item: str
    kind: str
    status: str
    title: Optional[str] = None
    source: Optional[str] = None
    landing_url: Optional[str] = None
    pdf_url: Optional[str] = None
    saved_path: Optional[str] = None
    note: Optional[str] = None


class Downloader:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.html_dir = self.out_dir / "html"
        self.pdf_dir = self.out_dir / "pdf"
        self.meta_dir = self.out_dir / "meta"
        for d in [self.out_dir, self.html_dir, self.pdf_dir, self.meta_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def fetch(self, url: str, *, pdf: bool = False, allow_redirects: bool = True) -> requests.Response:
        headers = PDF_HEADERS if pdf else HEADERS
        resp = self.session.get(url, headers=headers, timeout=TIMEOUT, allow_redirects=allow_redirects)
        resp.raise_for_status()
        return resp

    def save_binary(self, url: str, dest: Path) -> Path:
        with self.session.get(url, headers=PDF_HEADERS, timeout=TIMEOUT, stream=True) as r:
            r.raise_for_status()
            ctype = (r.headers.get("Content-Type") or "").lower()
            if "pdf" not in ctype and not url.lower().endswith(".pdf"):
                raise ValueError(f"URL does not look like a PDF: {url} (content-type={ctype})")
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 128):
                    if chunk:
                        f.write(chunk)
        return dest

    def save_text(self, text: str, dest: Path) -> Path:
        dest.write_text(text, encoding="utf-8")
        return dest


def slugify(text: str, max_len: int = 120) -> str:
    text = text.strip().lower()
    text = re.sub(r"https?://", "", text)
    text = re.sub(r"[^\w\-\. ]+", "", text)
    text = re.sub(r"[\s/]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-._")
    return (text[:max_len] or "item")


def is_url(s: str) -> bool:
    try:
        p = urlparse(s)
        return p.scheme in {"http", "https"} and bool(p.netloc)
    except Exception:
        return False


def normalize_title(title: str) -> str:
    title = re.sub(r"\s+", " ", title).strip().lower()
    title = re.sub(r"[^a-z0-9: -]", "", title)
    return title


def title_match_score(query: str, candidate: str) -> float:
    q = normalize_title(query)
    c = normalize_title(candidate)
    if not q or not c:
        return 0.0
    if q == c:
        return 1.0
    if q in c or c in q:
        return 0.92
    q_tokens = set(q.split())
    c_tokens = set(c.split())
    if not q_tokens or not c_tokens:
        return 0.0
    inter = len(q_tokens & c_tokens)
    union = len(q_tokens | c_tokens)
    return inter / union


def discover_pdf_links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    candidates: list[str] = []

    for tag in soup.find_all(["a", "link", "meta"]):
        url = None
        if tag.name == "a":
            href = tag.get("href")
            text = (tag.get_text(" ", strip=True) or "").lower()
            if href and ("pdf" in href.lower() or "pdf" in text or "paper" in text or "download" in text):
                url = href
        elif tag.name == "link":
            href = tag.get("href")
            typ = (tag.get("type") or "").lower()
            rel = " ".join(tag.get("rel") or []).lower()
            if href and ("pdf" in typ or "alternate" in rel or href.lower().endswith(".pdf")):
                url = href
        elif tag.name == "meta":
            prop = (tag.get("property") or tag.get("name") or "").lower()
            content = tag.get("content")
            if content and ("citation_pdf_url" in prop or "pdf" in content.lower()):
                url = content
        if url:
            abs_url = urljoin(base_url, url)
            if abs_url not in candidates:
                candidates.append(abs_url)

    # Common patterns
    for pat in [r'https?://[^"\'\s>]+\.pdf(?:\?[^"\'\s>]*)?', r'"(\/[^"\']+\.pdf(?:\?[^"\']*)?)"']:
        for m in re.finditer(pat, html, flags=re.I):
            raw = m.group(0).strip('"')
            abs_url = urljoin(base_url, raw)
            if abs_url not in candidates:
                candidates.append(abs_url)

    return candidates


def try_download_pdf(dl: Downloader, pdf_url: str, stem: str) -> Optional[Path]:
    try:
        dest = dl.pdf_dir / f"{stem}.pdf"
        return dl.save_binary(pdf_url, dest)
    except Exception:
        return None


def process_direct_url(dl: Downloader, item: str) -> Result:
    stem = slugify(item)
    try:
        resp = dl.fetch(item)
        ctype = (resp.headers.get("Content-Type") or "").lower()
        if "pdf" in ctype or item.lower().endswith(".pdf"):
            saved = dl.pdf_dir / f"{stem}.pdf"
            dl.save_binary(resp.url, saved)
            return Result(item=item, kind="url", status="ok", source="direct", landing_url=item, pdf_url=resp.url, saved_path=str(saved))

        html_path = dl.html_dir / f"{stem}.html"
        dl.save_text(resp.text, html_path)

        pdf_links = discover_pdf_links(resp.text, resp.url)
        for pdf_url in pdf_links:
            saved = try_download_pdf(dl, pdf_url, stem)
            if saved:
                return Result(
                    item=item,
                    kind="url",
                    status="ok",
                    source="direct+discovered_pdf",
                    landing_url=resp.url,
                    pdf_url=pdf_url,
                    saved_path=str(saved),
                    note=f"HTML also saved to {html_path}",
                )

        return Result(
            item=item,
            kind="url",
            status="partial",
            source="direct",
            landing_url=resp.url,
            saved_path=str(html_path),
            note="Downloaded page HTML, but no PDF link was found.",
        )
    except Exception as e:
        return Result(item=item, kind="url", status="error", note=str(e))


def search_openalex(title: str) -> Optional[dict]:
    url = "https://api.openalex.org/works"
    params = {"search": title, "per-page": 10}
    resp = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()
    results = resp.json().get("results", [])
    best = None
    best_score = 0.0
    for r in results:
        cand_title = r.get("title") or ""
        score = title_match_score(title, cand_title)
        if score > best_score:
            best_score = score
            best = r
    if best and best_score >= 0.55:
        return best
    return None


def search_arxiv(title: str) -> Optional[dict]:
    query = f'ti:"{title}"'
    url = "https://export.arxiv.org/api/query"
    params = {"search_query": query, "start": 0, "max_results": 5}
    resp = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()
    feed = feedparser.parse(resp.text)
    best = None
    best_score = 0.0
    for entry in feed.entries:
        cand_title = entry.get("title", "")
        score = title_match_score(title, cand_title)
        if score > best_score:
            best_score = score
            best = entry
    if best and best_score >= 0.55:
        return {
            "title": best.get("title"),
            "landing_url": best.get("id"),
            "pdf_url": next((l.href for l in best.get("links", []) if getattr(l, "type", "") == "application/pdf"), None),
            "source": "arxiv",
        }
    return None


def search_crossref(title: str) -> Optional[dict]:
    url = "https://api.crossref.org/works"
    params = {"query.title": title, "rows": 10}
    resp = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()
    items = resp.json().get("message", {}).get("items", [])
    best = None
    best_score = 0.0
    for item in items:
        cand_title = " ".join(item.get("title", []))
        score = title_match_score(title, cand_title)
        if score > best_score:
            best_score = score
            best = item
    if best and best_score >= 0.55:
        doi = best.get("DOI")
        landing = f"https://doi.org/{doi}" if doi else None
        return {
            "title": " ".join(best.get("title", [])),
            "landing_url": landing,
            "doi": doi,
            "source": "crossref",
        }
    return None


def resolve_from_openalex(work: dict) -> dict:
    best_oa = work.get("best_oa_location") or {}
    primary = work.get("primary_location") or {}
    pdf_url = best_oa.get("pdf_url") or primary.get("pdf_url")
    landing = best_oa.get("landing_page_url") or primary.get("landing_page_url") or work.get("id")
    return {
        "title": work.get("title"),
        "landing_url": landing,
        "pdf_url": pdf_url,
        "source": "openalex",
    }


def process_title(dl: Downloader, item: str) -> Result:
    stem = slugify(item)
    errors = []

    # 1) OpenAlex
    try:
        work = search_openalex(item)
        if work:
            info = resolve_from_openalex(work)
            if info.get("pdf_url"):
                saved = try_download_pdf(dl, info["pdf_url"], stem)
                if saved:
                    return Result(item=item, kind="title", status="ok", title=info["title"], source=info["source"], landing_url=info["landing_url"], pdf_url=info["pdf_url"], saved_path=str(saved))
            if info.get("landing_url"):
                r = process_direct_url(dl, info["landing_url"])
                if r.status in {"ok", "partial"}:
                    r.item = item
                    r.kind = "title"
                    r.title = info["title"]
                    r.source = f"{info['source']}->{r.source}"
                    return r
    except Exception as e:
        errors.append(f"OpenAlex: {e}")
    time.sleep(SLEEP_SECONDS)

    # 2) arXiv
    try:
        info = search_arxiv(item)
        if info:
            if info.get("pdf_url"):
                saved = try_download_pdf(dl, info["pdf_url"], stem)
                if saved:
                    return Result(item=item, kind="title", status="ok", title=info["title"], source=info["source"], landing_url=info["landing_url"], pdf_url=info["pdf_url"], saved_path=str(saved))
            if info.get("landing_url"):
                r = process_direct_url(dl, info["landing_url"])
                if r.status in {"ok", "partial"}:
                    r.item = item
                    r.kind = "title"
                    r.title = info["title"]
                    r.source = f"{info['source']}->{r.source}"
                    return r
    except Exception as e:
        errors.append(f"arXiv: {e}")
    time.sleep(SLEEP_SECONDS)

    # 3) Crossref
    try:
        info = search_crossref(item)
        if info and info.get("landing_url"):
            r = process_direct_url(dl, info["landing_url"])
            if r.status in {"ok", "partial"}:
                r.item = item
                r.kind = "title"
                r.title = info["title"]
                r.source = f"{info['source']}->{r.source}"
                return r
    except Exception as e:
        errors.append(f"Crossref: {e}")

    return Result(item=item, kind="title", status="error", note="; ".join(errors) or "No result found")


def load_items(path: Optional[str]) -> list[str]:
    if not path:
        return DEFAULT_ITEMS[:]
    content = Path(path).read_text(encoding="utf-8")
    return [line.strip() for line in content.splitlines() if line.strip()]


def write_reports(out_dir: Path, results: list[Result]) -> None:
    meta_path = out_dir / "metadata.json"
    meta_path.write_text(json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = out_dir / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["item", "kind", "status", "title", "source", "landing_url", "pdf_url", "saved_path", "note"],
        )
        w.writeheader()
        for r in results:
            w.writerow(asdict(r))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="downloads_video_action", help="Output directory")
    parser.add_argument("--items", help="Text file with one URL/title per line")
    args = parser.parse_args()

    out_dir = Path(args.out).resolve()
    items = load_items(args.items)
    dl = Downloader(out_dir)

    print(f"[INFO] Will process {len(items)} items into: {out_dir}")
    results: list[Result] = []
    for idx, item in enumerate(items, 1):
        print(f"[{idx}/{len(items)}] {item}")
        if is_url(item):
            res = process_direct_url(dl, item)
        else:
            res = process_title(dl, item)
        results.append(res)
        print(f"    -> {res.status} | source={res.source} | saved={res.saved_path}")
        time.sleep(SLEEP_SECONDS)

    write_reports(out_dir, results)

    ok = sum(r.status == "ok" for r in results)
    partial = sum(r.status == "partial" for r in results)
    err = sum(r.status == "error" for r in results)
    print(f"[DONE] ok={ok}, partial={partial}, error={err}")
    print(f"[DONE] metadata: {out_dir / 'metadata.json'}")
    print(f"[DONE] summary : {out_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
