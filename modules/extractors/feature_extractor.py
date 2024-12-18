def get_added_line_count(filename: str, commits: list[dict]) -> int:
    added_line_count = 0
    for commit in commits:
        for filepath, stats in commit["files"].items():
            if filepath.endswith(filename):
                added_line_count += stats["insertions"]
    return added_line_count


def get_deleted_line_count(filename: str, commits: list[dict]) -> int:
    deleted_line_count = 0
    for commit in commits:
        for filepath, stats in commit["files"].items():
            if filepath.endswith(filename):
                deleted_line_count += stats["deletions"]
    return deleted_line_count


def get_serverities(commits: list[dict], bugs: list[dict]) -> list[str]:
    serverities = []
    for commit in commits:
        for bug in bugs:
            if commit["message"].startswith(bug["key"]):
                serverities.append(bug["priority"])
    return serverities


def get_dev_count(commits: list[dict]) -> int:
    devs = set()
    for commit in commits:
        devs.add(commit["author"])
    return len(devs)


def get_avg_time_between_commit(commits: list[dict]) -> float | None:
    if len(commits) < 2:
        return None
    time = 0
    for i in range(1, len(commits)):
        time += commits[i]["committed_date"] - commits[i - 1]["committed_date"]
    return time / (len(commits) - 1)


def get_avg_dev_expertise(commits: list[dict], devs: list[str]) -> float | None:
    devs_expertise = {}
    for commit in commits:
        dev = commit["author"] 
        if dev in devs:
            devs_expertise.setdefault(dev, 0)
            devs_expertise[dev] += 1
    if len(devs_expertise) == 0:
        return None
    return sum(devs_expertise.values()) / len(devs_expertise)


def get_min_dev_expertise(commits: list[dict], devs: list[str]) -> int:
    devs_expertise = {}
    for commit in commits:
        dev = commit["author"] 
        if dev in devs:
            devs_expertise.setdefault(dev, 0)
            devs_expertise[dev] += 1
    if len(devs_expertise) == 0:
        return None
    return min(devs_expertise.values())


def _is_comment(line: str) -> bool:
    stripped_line = line.strip()
    if not stripped_line:
        return False
    return stripped_line.startswith("//") or stripped_line.startswith("/*") or stripped_line.startswith("*")


def  _commit_modified_comment(filepath: str, commit: dict) -> bool:
    diffs = commit["files"][filepath]["diff"]
    for diff in diffs:
        for line in diff.split("\n"):
            if (line.startswith('+') or line.startswith('-')) and _is_comment(line[1:]):
                return True
    return False


def get_commit_modified_comment_count(filename: str, commits: list[dict]) -> int:
    commit_count = 0
    for commit in commits:
        filepath = ""
        for e in commit["files"].keys():
            if e.endswith(filename):
                filepath = e
                break
        if _commit_modified_comment(filepath, commit):
            commit_count += 1
    return commit_count
