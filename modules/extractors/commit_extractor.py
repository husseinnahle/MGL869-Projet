from concurrent.futures import ThreadPoolExecutor
import sys
from git import Commit

from modules.utils.git_repo import GitRepo


def _process_commit(commit: Commit, diff: bool) -> dict:
	return {
		"author": commit.author.email,
		"files": {
			filepath:
			{
				"insertions": stats["insertions"],
				"deletions": stats["deletions"],
				"diff": [] if not diff else [e.diff.decode("utf-8") for e in commit.diff(commit.parents[0], paths=filepath, create_patch=True) if e.diff]
			}
			for filepath, stats in commit.stats.files.items() if filepath.endswith(".java")
		},
		"message": commit.message,
		"committed_date": commit.committed_date
	}


def extract_commits(rev: str, diff: bool):
	repo = GitRepo("hive", "apache", "https://github.com", "C:\\Users\\husse\\root\\MGL869")
	repo.clone()
	raw_commits = repo.get_commits(rev)
	commits = []
	for i, raw_commit in enumerate(raw_commits):
		print(f"\rProcessing commit {i + 1}/{len(raw_commits)}", end="")
		sys.stdout.flush()
		commits.append(_process_commit(raw_commit, diff))
	print("\n")
	return sorted(commits, key=lambda x: x["committed_date"])

