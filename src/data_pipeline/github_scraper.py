"""
GitHub KiCad Project Scraper
============================
Search GitHub for open-source KiCad projects containing .kicad_sch files,
clone them, and collect schematic files for dataset construction.

Key features:
- GitHub Search API for discovering repositories
- Shallow clone for bandwidth efficiency
- Filtering by file size and component count
- Focus on MCU/sensor/power projects (differentiation strategy)
"""

import os
import json
import time
import logging
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ScrapedProject:
    """Metadata for a scraped KiCad project."""
    repo_name: str
    repo_url: str
    clone_path: str
    sch_files: List[str] = field(default_factory=list)
    star_count: int = 0
    description: str = ""


class GitHubKiCadScraper:
    """Scrapes GitHub for open-source KiCad schematic projects."""

    # Search queries targeting different circuit categories
    SEARCH_QUERIES = [
        # General KiCad schematics
        "extension:kicad_sch size:>1000",
        "extension:kicad_sch STM32",
        "extension:kicad_sch ESP32",
        "extension:kicad_sch RP2040",
        "extension:kicad_sch Arduino",
        # Sensor circuits
        "extension:kicad_sch sensor",
        "extension:kicad_sch I2C SPI",
        # Power circuits
        "extension:kicad_sch power supply",
        "extension:kicad_sch buck boost regulator",
        "extension:kicad_sch LDO PMIC",
        # Communication interfaces
        "extension:kicad_sch RS485 CAN USB",
        "extension:kicad_sch ethernet PHY",
        # FPGA / digital
        "extension:kicad_sch FPGA",
        "extension:kicad_sch CPLD",
        # Audio / analog
        "extension:kicad_sch amplifier opamp",
        "extension:kicad_sch audio codec",
    ]

    def __init__(
        self,
        output_dir: str,
        github_token: Optional[str] = None,
        max_repos: int = 300,
        min_file_size: int = 1000,
    ):
        """
        Args:
            output_dir: Directory to clone repos into
            github_token: GitHub personal access token (optional but recommended)
            max_repos: Maximum number of repos to clone
            min_file_size: Minimum .kicad_sch file size in bytes
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.github_token = github_token or os.environ.get("GITHUB_TOKEN")
        self.max_repos = max_repos
        self.min_file_size = min_file_size
        self.session_repos: Dict[str, ScrapedProject] = {}
        self.metadata_file = self.output_dir / "scraped_metadata.json"

    def _make_headers(self) -> Dict[str, str]:
        """Build HTTP headers with optional auth."""
        headers = {"Accept": "application/vnd.github.v3+json"}
        if self.github_token:
            headers["Authorization"] = f"token {self.github_token}"
        return headers

    def search_repos(self, per_page: int = 100, pages: int = 5) -> List[Dict]:
        """Search GitHub for repositories containing .kicad_sch files."""
        import requests

        all_items = []
        headers = self._make_headers()

        for query in self.SEARCH_QUERIES:
            for page in range(1, pages + 1):
                url = "https://api.github.com/search/code"
                params = {
                    "q": query,
                    "per_page": per_page,
                    "page": page,
                }

                try:
                    resp = requests.get(url, headers=headers, params=params)

                    # Rate limiting
                    if resp.status_code == 403:
                        reset_time = int(resp.headers.get("X-RateLimit-Reset", 0))
                        wait = max(reset_time - time.time(), 60)
                        logger.warning(f"Rate limited. Waiting {wait:.0f}s...")
                        time.sleep(wait)
                        continue

                    if resp.status_code != 200:
                        logger.warning(f"Search failed ({resp.status_code}): {query}")
                        break

                    data = resp.json()
                    items = data.get("items", [])
                    all_items.extend(items)

                    logger.info(
                        f"Query '{query}' page {page}: found {len(items)} items"
                    )

                    # Respect rate limits
                    remaining = int(resp.headers.get("X-RateLimit-Remaining", 1000))
                    if remaining < 10:
                        time.sleep(30)
                    else:
                        time.sleep(2)

                except Exception as e:
                    logger.error(f"Search error: {e}")
                    break

        # Deduplicate by repo
        seen_repos = {}
        for item in all_items:
            repo_info = item.get("repository", {})
            repo_name = repo_info.get("full_name", "")
            if repo_name and repo_name not in seen_repos:
                seen_repos[repo_name] = {
                    "full_name": repo_name,
                    "clone_url": repo_info.get("clone_url", ""),
                    "html_url": repo_info.get("html_url", ""),
                    "stargazers_count": repo_info.get("stargazers_count", 0),
                    "description": repo_info.get("description", ""),
                    "sch_path": item.get("path", ""),
                }

        logger.info(f"Found {len(seen_repos)} unique repositories")
        return list(seen_repos.values())[:self.max_repos]

    def clone_repo(self, repo_info: Dict) -> Optional[ScrapedProject]:
        """Shallow-clone a repository."""
        repo_name = repo_info["full_name"]
        safe_name = repo_name.replace("/", "__")
        clone_path = self.output_dir / safe_name

        if clone_path.exists():
            logger.info(f"Already cloned: {repo_name}")
            project = ScrapedProject(
                repo_name=repo_name,
                repo_url=repo_info.get("html_url", ""),
                clone_path=str(clone_path),
                star_count=repo_info.get("stargazers_count", 0),
                description=repo_info.get("description", ""),
            )
            self._find_sch_files(project)
            return project

        try:
            cmd = [
                "git", "clone",
                "--depth", "1",
                "--single-branch",
                repo_info.get("clone_url", ""),
                str(clone_path),
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120
            )

            if result.returncode != 0:
                logger.warning(f"Clone failed for {repo_name}: {result.stderr[:200]}")
                return None

            project = ScrapedProject(
                repo_name=repo_name,
                repo_url=repo_info.get("html_url", ""),
                clone_path=str(clone_path),
                star_count=repo_info.get("stargazers_count", 0),
                description=repo_info.get("description", ""),
            )
            self._find_sch_files(project)
            return project

        except subprocess.TimeoutExpired:
            logger.warning(f"Clone timeout for {repo_name}")
            return None
        except Exception as e:
            logger.error(f"Clone error for {repo_name}: {e}")
            return None

    def _find_sch_files(self, project: ScrapedProject):
        """Find all .kicad_sch files in a cloned project."""
        clone_path = Path(project.clone_path)
        if not clone_path.exists():
            return

        sch_files = []
        for sch_file in clone_path.rglob("*.kicad_sch"):
            if sch_file.stat().st_size >= self.min_file_size:
                sch_files.append(str(sch_file))

        project.sch_files = sch_files
        logger.info(
            f"Found {len(sch_files)} schematic files in {project.repo_name}"
        )

    def run(self) -> List[ScrapedProject]:
        """Execute the full scraping pipeline."""
        logger.info("Starting GitHub KiCad project scraping...")

        # Search
        repos = self.search_repos()
        logger.info(f"Found {len(repos)} repositories to clone")

        # Clone
        projects = []
        for i, repo_info in enumerate(repos):
            logger.info(f"Cloning {i+1}/{len(repos)}: {repo_info['full_name']}")
            project = self.clone_repo(repo_info)
            if project and project.sch_files:
                projects.append(project)
                self.session_repos[project.repo_name] = project

        # Save metadata
        self._save_metadata(projects)

        logger.info(
            f"Scraping complete: {len(projects)} projects with "
            f"{sum(len(p.sch_files) for p in projects)} schematic files"
        )
        return projects

    def _save_metadata(self, projects: List[ScrapedProject]):
        """Save scraping metadata to JSON."""
        metadata = []
        for p in projects:
            metadata.append({
                "repo_name": p.repo_name,
                "repo_url": p.repo_url,
                "clone_path": p.clone_path,
                "sch_files": p.sch_files,
                "star_count": p.star_count,
                "description": p.description,
            })

        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"Metadata saved to {self.metadata_file}")

    def load_metadata(self) -> List[ScrapedProject]:
        """Load previously scraped metadata."""
        if not self.metadata_file.exists():
            return []

        with open(self.metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        projects = []
        for m in metadata:
            p = ScrapedProject(**m)
            if Path(p.clone_path).exists():
                projects.append(p)

        return projects


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Scrape GitHub for KiCad projects")
    parser.add_argument("--output-dir", default="data/raw", help="Output directory")
    parser.add_argument("--token", default=None, help="GitHub token")
    parser.add_argument("--max-repos", type=int, default=100)
    args = parser.parse_args()

    scraper = GitHubKiCadScraper(
        output_dir=args.output_dir,
        github_token=args.token,
        max_repos=args.max_repos,
    )
    projects = scraper.run()
    print(f"Scraped {len(projects)} projects")
