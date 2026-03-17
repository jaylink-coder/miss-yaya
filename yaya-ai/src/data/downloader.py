"""Dataset download orchestrator for Yaya AI data engine.

Handles parallel downloading of datasets from HuggingFace Hub,
direct URLs, and local file registration. Supports resumable
downloads and progress tracking.
"""

import os
import json
import time
import hashlib
import subprocess
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml


@dataclass
class DatasetInfo:
    """Metadata for a single dataset source."""
    name: str
    source: str              # huggingface, url, local
    category: str            # web_text, code, books, etc.
    format: str              # jsonl, parquet, txt, json
    text_field: str = "text"
    splits: List[str] = field(default_factory=lambda: ["train"])
    subset: Optional[str] = None
    description: str = ""
    priority: int = 1
    filters: Dict[str, Any] = field(default_factory=dict)
    url: Optional[str] = None
    local_path: Optional[str] = None


class DownloadManager:
    """Orchestrates dataset downloads and tracks progress.

    Supports:
    - HuggingFace Hub datasets (streaming or full download)
    - Direct URL downloads
    - Local file registration
    - Resumable downloads with manifest tracking
    """

    def __init__(
        self,
        output_dir: str = "data/raw",
        cache_dir: str = "data/cache",
        max_workers: int = 4,
    ):
        self.output_dir = output_dir
        self.cache_dir = cache_dir
        self.max_workers = max_workers
        self.manifest_path = os.path.join(output_dir, "download_manifest.json")
        self.manifest = self._load_manifest()

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)

    def _load_manifest(self) -> Dict[str, Any]:
        """Load or create the download tracking manifest."""
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path, "r") as f:
                return json.load(f)
        return {"datasets": {}, "last_updated": None}

    def _save_manifest(self):
        """Persist the manifest to disk."""
        self.manifest["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        os.makedirs(os.path.dirname(self.manifest_path), exist_ok=True)
        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2)

    @staticmethod
    def load_sources_config(config_path: str) -> Dict[str, DatasetInfo]:
        """Load dataset definitions from sources.yaml config."""
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        datasets = {}
        for key, raw in config.get("datasets", {}).items():
            datasets[key] = DatasetInfo(
                name=raw.get("name", key),
                source=raw.get("source", "huggingface"),
                category=raw.get("category", "web_text"),
                format=raw.get("format", "jsonl"),
                text_field=raw.get("text_field", "text"),
                splits=raw.get("splits", ["train"]),
                subset=raw.get("subset"),
                description=raw.get("description", ""),
                priority=raw.get("priority", 1),
                filters=raw.get("filters", {}),
                url=raw.get("url"),
                local_path=raw.get("local_path"),
            )
        return datasets

    def get_dataset_dir(self, dataset_key: str, category: str) -> str:
        """Get the output directory for a dataset."""
        path = os.path.join(self.output_dir, category, dataset_key)
        os.makedirs(path, exist_ok=True)
        return path

    def is_downloaded(self, dataset_key: str) -> bool:
        """Check if a dataset has already been downloaded."""
        entry = self.manifest["datasets"].get(dataset_key, {})
        return entry.get("status") == "complete"

    def download_huggingface(
        self,
        dataset_key: str,
        info: DatasetInfo,
        max_samples: Optional[int] = None,
        streaming: bool = True,
    ) -> str:
        """Download a dataset from HuggingFace Hub.

        Args:
            dataset_key: Unique key for this dataset.
            info: Dataset metadata.
            max_samples: Limit number of samples (for testing).
            streaming: Use streaming mode (memory efficient).

        Returns:
            Path to the downloaded data directory.
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "HuggingFace datasets required: pip install datasets"
            )

        output_dir = self.get_dataset_dir(dataset_key, info.category)
        print(f"  Downloading {info.name} -> {output_dir}")

        self.manifest["datasets"][dataset_key] = {
            "name": info.name,
            "category": info.category,
            "status": "downloading",
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "output_dir": output_dir,
        }
        self._save_manifest()

        try:
            for split in info.splits:
                kwargs = {"split": split, "streaming": streaming}
                if info.subset:
                    kwargs["name"] = info.subset

                ds = load_dataset(
                    info.name,
                    cache_dir=self.cache_dir,
                    **kwargs,
                )

                output_file = os.path.join(output_dir, f"{split}.jsonl")
                count = 0

                with open(output_file, "w", encoding="utf-8") as f:
                    for sample in ds:
                        text = sample.get(info.text_field, "")
                        if not text or not text.strip():
                            continue

                        # Apply dataset-specific filters
                        if info.filters:
                            if "languages" in info.filters:
                                lang = sample.get("language", sample.get("lang", ""))
                                if lang and lang not in info.filters["languages"]:
                                    continue

                        record = {"text": text, "source": dataset_key}
                        if "language" in sample:
                            record["language"] = sample["language"]

                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        count += 1

                        if count % 100000 == 0:
                            print(f"    {dataset_key}/{split}: {count:,} docs")

                        if max_samples and count >= max_samples:
                            break

                print(f"    {dataset_key}/{split}: {count:,} docs total")

            self.manifest["datasets"][dataset_key]["status"] = "complete"
            self.manifest["datasets"][dataset_key]["completed"] = time.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            self._save_manifest()

        except Exception as e:
            self.manifest["datasets"][dataset_key]["status"] = "failed"
            self.manifest["datasets"][dataset_key]["error"] = str(e)
            self._save_manifest()
            raise

        return output_dir

    def download_url(self, dataset_key: str, info: DatasetInfo) -> str:
        """Download a dataset from a direct URL."""
        import urllib.request

        output_dir = self.get_dataset_dir(dataset_key, info.category)
        url = info.url
        if not url:
            raise ValueError(f"No URL specified for dataset {dataset_key}")

        filename = url.split("/")[-1]
        output_path = os.path.join(output_dir, filename)

        print(f"  Downloading {url} -> {output_path}")

        self.manifest["datasets"][dataset_key] = {
            "name": info.name,
            "category": info.category,
            "status": "downloading",
            "url": url,
            "output_dir": output_dir,
        }
        self._save_manifest()

        try:
            urllib.request.urlretrieve(url, output_path)
            self.manifest["datasets"][dataset_key]["status"] = "complete"
            self._save_manifest()
        except Exception as e:
            self.manifest["datasets"][dataset_key]["status"] = "failed"
            self.manifest["datasets"][dataset_key]["error"] = str(e)
            self._save_manifest()
            raise

        return output_dir

    def register_local(self, dataset_key: str, info: DatasetInfo) -> str:
        """Register a local dataset (no download needed)."""
        local_path = info.local_path
        if not local_path or not os.path.exists(local_path):
            raise ValueError(f"Local path not found: {local_path}")

        self.manifest["datasets"][dataset_key] = {
            "name": info.name,
            "category": info.category,
            "status": "complete",
            "output_dir": local_path,
        }
        self._save_manifest()
        print(f"  Registered local: {dataset_key} -> {local_path}")
        return local_path

    def download_dataset(
        self,
        dataset_key: str,
        info: DatasetInfo,
        max_samples: Optional[int] = None,
        force: bool = False,
    ) -> str:
        """Download a single dataset using the appropriate method.

        Args:
            dataset_key: Unique identifier for the dataset.
            info: Dataset metadata.
            max_samples: Limit samples (for testing).
            force: Re-download even if already complete.

        Returns:
            Path to downloaded data.
        """
        if not force and self.is_downloaded(dataset_key):
            print(f"  Skipping {dataset_key} (already downloaded)")
            return self.manifest["datasets"][dataset_key]["output_dir"]

        if info.source == "huggingface":
            return self.download_huggingface(dataset_key, info, max_samples)
        elif info.source == "url":
            return self.download_url(dataset_key, info)
        elif info.source == "local":
            return self.register_local(dataset_key, info)
        else:
            raise ValueError(f"Unknown source type: {info.source}")

    def download_all(
        self,
        datasets: Dict[str, DatasetInfo],
        categories: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
        force: bool = False,
        parallel: bool = False,
    ) -> Dict[str, str]:
        """Download all configured datasets.

        Args:
            datasets: Dict of dataset_key -> DatasetInfo.
            categories: Only download these categories (None = all).
            max_samples: Limit samples per dataset (for testing).
            force: Re-download all.
            parallel: Use parallel downloads.

        Returns:
            Dict of dataset_key -> output_dir.
        """
        # Filter by category and sort by priority
        filtered = {}
        for key, info in datasets.items():
            if categories and info.category not in categories:
                continue
            filtered[key] = info

        sorted_keys = sorted(filtered.keys(), key=lambda k: filtered[k].priority)

        print(f"\nDownloading {len(sorted_keys)} datasets:")
        for key in sorted_keys:
            info = filtered[key]
            print(f"  [{info.priority}] {key}: {info.name} ({info.category})")
        print()

        results = {}

        if parallel and len(sorted_keys) > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {}
                for key in sorted_keys:
                    future = executor.submit(
                        self.download_dataset, key, filtered[key], max_samples, force
                    )
                    futures[future] = key

                for future in as_completed(futures):
                    key = futures[future]
                    try:
                        results[key] = future.result()
                    except Exception as e:
                        print(f"  FAILED: {key}: {e}")
        else:
            for key in sorted_keys:
                try:
                    results[key] = self.download_dataset(
                        key, filtered[key], max_samples, force
                    )
                except Exception as e:
                    print(f"  FAILED: {key}: {e}")

        # Summary
        complete = sum(
            1 for k, v in self.manifest["datasets"].items() if v.get("status") == "complete"
        )
        failed = sum(
            1 for k, v in self.manifest["datasets"].items() if v.get("status") == "failed"
        )
        print(f"\nDownload summary: {complete} complete, {failed} failed")

        return results

    def status(self) -> Dict[str, Any]:
        """Return current download status for all datasets."""
        summary = {"complete": [], "downloading": [], "failed": [], "pending": []}
        for key, entry in self.manifest.get("datasets", {}).items():
            status = entry.get("status", "unknown")
            summary.get(status, summary["pending"]).append(key)
        return summary
