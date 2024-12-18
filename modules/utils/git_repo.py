import os
from pathlib import Path
from git import Commit, Repo


class GitRepo:

    def __init__(self, name: str, owner: str, url: str, base_dir: str):
        self.name = name
        self.owner = owner
        self.url = url
        self.base_dir = Path(base_dir)
        self.repo = self._get_repo()

    def _get_repo(self) -> Repo | None:
        path = self.get_path()
        if os.path.exists(path):
            return Repo(path)
        return None

    def get_path(self) -> str:
        return (self.base_dir / self.name).absolute().as_posix()

    def get_url(self) -> str:
        return f"{self.url}/{self.owner}/{self.name}"

    def get_commits(self, rev: str = "--all") -> list[Commit]:
        return list(self.repo.iter_commits(rev))

    def clone(self) -> None:
        if not self.repo:
            url = self.get_url() + ".git"
            path = self.get_path()
            self.repo = Repo.clone_from(url, path, no_checkout=True)

    def checkout(self, sha: str) -> None:
        self.repo.git.checkout(sha)
