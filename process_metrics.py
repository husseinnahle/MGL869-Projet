from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import sys
import pandas as pd

from modules.extractors.commit_extractor import extract_commits
from modules.extractors.feature_extractor import get_added_line_count, get_avg_dev_expertise, get_avg_time_between_commit, get_commit_modified_comment_count, get_serverities, \
  get_deleted_line_count, get_dev_count, get_min_dev_expertise


CLASS_METRICS = ["CountClassBase", "CountClassCoupled", "CountClassDerived", "MaxInheritanceTree", "PercentLackOfCohesion"]
METHOD_METRICS = ["CountInput", "CountOutput", "CountPath", "MaxNesting"]
OTHER_METRICS = [
    "Kind", "CCViolDensityCode", "CCViolDensityLine", "CountCCViol", "CountCCViolType", "CountClassCoupledModified",
    "CountDeclExecutableUnit", "CountDeclFile", "CountDeclMethodAll", "Cyclomatic", "PercentLackOfCohesionModified"
]
NEW_METRICS = [
    "AddedLineCount", "DeletedLineCount", "CurrCommitCount", "BugFixCount", "AllCommitCount",
    "CurrDevCount", "AllDevCount", "AvgTimeBetweenCurrCommit", "AvgTimeBetweenAllCommit",
    "AvgDevExpertise", "MinDevExpertise", "CommitModifiedCommentCount", "CommitNotModifiedCommentCount", "BugSeverity"
]
ALL_METRICS = [
	"AvgCountLine", "AvgCountLineBlank", "AvgCountLineCode", "AvgCountLineComment", "AvgCyclomatic", "AvgCyclomaticModified", "AvgCyclomaticStrict",
	"AvgEssential", "CountDeclClass", "CountDeclClassMethod", "CountDeclClassVariable", "CountDeclFunction", "CountDeclInstanceMethod",
	"CountDeclInstanceVariable", "CountDeclMethod", "CountDeclMethodDefault", "CountDeclMethodPrivate", "CountDeclMethodProtected", "CountDeclMethodPublic",
	"CountLine", "CountLineBlank", "CountLineCode", "CountLineCodeDecl", "CountLineCodeExe", "CountLineComment", "CountSemicolon", "CountStmt",
	"CountStmtDecl", "CountStmtExe", "MaxCyclomatic", "MaxCyclomaticModified", "MaxCyclomaticStrict", "RatioCommentToCode", "SumCyclomatic", "SumCyclomaticModified",
	"SumCyclomaticStrict", "SumEssential", "MinCountInput","MaxCountInput", "MeanCountInput", "MinCountOutput","MaxCountOutput", "MeanCountOutput", "MinCountPath",
	"MaxCountPath", "MeanCountPath", "MinMaxNesting", "MiaxaxNesting", "MeanMaxNesting", "MeanCountClassBase", "MeanCountClassCoupled",
	"MeanCountClassDerived", "MeanMaxInheritanceTree", "MeanPercentLackOfCohesion"
] + NEW_METRICS
SEVERITY_MAP = {
    "NOT_A_BUG": 0,
    "Trivial": 1,
    "Minor": 2,
    "Major": 3,
    "Critical": 4,
    "Blocker": 5
}
BASE_DIR = "C:\\Users\\husse\\root\\MGL869\\MGL869-Projet"
METRICS_FILE = BASE_DIR + "\\data\\metrics\\und_hive_%s.csv"
BUGS_FILE = BASE_DIR + "\\data\\issues.json"
VERSIONS = {
	"2.0.0": "7f9f1fcb8697fb33f0edc2c391930a3728d247d7",
	"2.1.0": "e3cfeebcefe9a19c5055afdcbb00646908340694",
	"2.2.0": "da840b0f8fa99cab9f004810cd22abc207493cae",
	"2.3.0": "6f4c35c9e904d226451c465effdc5bfd31d395a0",
	"3.0.0": "ce61711a5fa54ab34fc74d86d521ecaeea6b072a",
	"3.1.0": "bcc7df95824831a8d2f1524e4048dfc23ab98c19"
}



def _get_min_max_mean(data: pd.DataFrame, column: str) -> tuple:
    """Calculer le min, max et la moyenne d'une colonne d'un dataset.

    Args:
        data (pd.DataFrame): Dataset.
        column (str): Nom de la colonne.

    Returns:
        tuple: (min, max, mean)
    """
    column_values = data[column].dropna().astype(float)
    return column_values.min(), column_values.max(), column_values.mean()


def _process_class_method_metrics(filename: str, methods_dataset: pd.DataFrame, classes_dataset: pd.DataFrame) -> tuple:
    """Calculer les métriques des methodes et des classes pour un fichier.

    Args:
        filename (str): Nom du fichier.
        methods_dataset (pd.DataFrame): Métriques des méthodes.
        classes_dataset (pd.DataFrame): Métriques des classes.

    Returns:
        tuple: (filename, Nouvelles métriques)
    """
    file_methods = methods_dataset[methods_dataset["Name"].str.contains(filename.removesuffix(".java"), regex=False)]
    file_classes = classes_dataset[classes_dataset["Name"].str.contains(filename.removesuffix(".java"), regex=False)]
    method_metrics = {}
    for metric in METHOD_METRICS:
        min_val, max_val, mean_val = _get_min_max_mean(file_methods, metric)
        method_metrics[f"Min{metric}"] = min_val
        method_metrics[f"Max{metric}"] = max_val
        method_metrics[f"Mean{metric}"] = mean_val    
    class_metrics = {}
    for metric in CLASS_METRICS:
        _, _, mean_val = _get_min_max_mean(file_classes, metric)
        class_metrics[f"Mean{metric}"] = mean_val
    return filename, {**method_metrics, **class_metrics}


def _process_metrics(dataset: pd.DataFrame) -> pd.DataFrame:
	"""Calculer les métriques des methodes et des classes pour un fichier.

	Args:
		dataset (pd.DataFrame): Dataset contenant les métriques Understand.

	Returns:
		pd.DataFrame: Dataset fomatté.
	"""
	# Récupérer les métriques des méthodes et des classes
	file_names = dataset[dataset["Kind"] == "File"]["Name"].unique()
	methods_dataset = dataset[dataset["Kind"].str.contains("Method", regex=False)]
	classes_dataset = dataset[dataset["Kind"].str.contains("Class", regex=False)]

	# Pour accélérer le traitement, faire du multithreading
	file_metrics = {}
	total_files = len(file_names)
	with ThreadPoolExecutor() as executor:
		futures = {
			executor.submit(_process_class_method_metrics, file_name, methods_dataset, classes_dataset): file_name
			for file_name in file_names
		}
		for i, future in enumerate(as_completed(futures)):
			file_name, metrics = future.result()
			file_metrics[file_name] = metrics
			print(f"\rProcessing file metrics {i + 1}/{total_files}", end="")
			sys.stdout.flush()

	print("\n")
	# Joindre les nouvelles métriques au dataset et supprimer les anciennes
	file_metrics_df = pd.DataFrame.from_dict(file_metrics, orient='index').reset_index()
	file_metrics_df.columns = ["Name"] + list(file_metrics_df.columns[1:])
	dataset = dataset[dataset["Kind"] == "File"].copy()
	dataset = dataset.merge(file_metrics_df, on="Name", how="left")
	return dataset.drop(columns=METHOD_METRICS + CLASS_METRICS + OTHER_METRICS)


def _process_new_metrics(filename, commits, bugs):
	current_commits = commits["current_commits"]
	previous_commits = commits["previous_commits"]
	severities = get_serverities(current_commits, bugs)
	if not current_commits:
		all_commits_count = len(previous_commits)
		all_dev_count = get_dev_count(previous_commits)
		avg_time_between_all_commit = get_avg_time_between_commit(previous_commits)
		return None, []
		# return filename, [0, 0, 0, 0, all_commits_count, 0, all_dev_count, None, avg_time_between_all_commit, None, None, 0, 0, 0]
	else:
		added_line_count = get_added_line_count(filename, current_commits)
		deleted_line_count = get_deleted_line_count(filename, current_commits)
		current_commits_count = len(current_commits)
		bug_fix_count = len(severities)
		all_commits_count = current_commits_count + len(previous_commits)
		current_dev_count = get_dev_count(current_commits)
		all_dev_count = get_dev_count(previous_commits + current_commits)
		avg_time_between_curr_commit = get_avg_time_between_commit(current_commits)
		avg_time_between_all_commit = get_avg_time_between_commit(previous_commits+current_commits)
		avg_dev_expertise = get_avg_dev_expertise(previous_commits, [commit["author"] for commit in current_commits])
		min_dev_expertise = get_min_dev_expertise(previous_commits, [commit["author"] for commit in current_commits])        
		commit_modified_comment_count = get_commit_modified_comment_count(filename, current_commits)
		commit_not_modified_comment_count = len(current_commits) - commit_modified_comment_count
		bug_severity = max([SEVERITY_MAP[e] for e in severities] or [0])
		return filename, [added_line_count, deleted_line_count, current_commits_count, bug_fix_count, all_commits_count,
			current_dev_count, all_dev_count, avg_time_between_curr_commit, avg_time_between_all_commit,
			avg_dev_expertise, min_dev_expertise, commit_modified_comment_count, commit_not_modified_comment_count, bug_severity]


def _add_new_metrics(dataset: pd.DataFrame, changed_files: dict, bugs: list) -> pd.DataFrame:
	new_metrics = {}
	total_files = len(changed_files)
	with ThreadPoolExecutor() as executor:
		futures = [executor.submit(_process_new_metrics, filename, commits, bugs) for filename, commits in changed_files.items()]
		for i, future in enumerate(as_completed(futures)):
			filename, metrics = future.result()
			if filename != None:
				new_metrics[filename] = metrics
			print(f"\rProcessing new metrics {i + 1}/{total_files}", end="")
			sys.stdout.flush()

	print("\n")
	new_metrics_df = pd.DataFrame.from_dict(new_metrics, orient='index', columns=NEW_METRICS).reset_index()
	new_metrics_df.rename(columns={'index': 'Name'}, inplace=True)
	return dataset.merge(new_metrics_df, on="Name", how="inner")


def get_metrics(to_release_version: str, actual_version: str) -> pd.DataFrame:
	"""Calculer les métriques Understand pour une version donnée.

	Args:
			to_release_version (str): Version cible.
			actual_version (str): Version actuelle.

	Returns:
			pd.DataFrame: Dataset contenant les métriques calculées.
	"""
	print("Reading issues file...")
	with open(BUGS_FILE) as writer:
		issues: dict = json.load(writer)
	bugs = [e for e in issues[actual_version]]
	del issues

	print("Processing Understand metrics...")
	dataset = pd.read_csv(METRICS_FILE % actual_version.replace(".", "_"))
	processed_dataset = _process_metrics(dataset)
	del dataset
	filenames = processed_dataset["Name"].values.tolist()

	print(f"Extracting commits for version {to_release_version}...")
	current_commits = extract_commits(VERSIONS[actual_version] + ".." + VERSIONS[to_release_version] + "^", True)

	print(f"Extracting commits for all versions before {actual_version}...")
	previous_commits = extract_commits(".." + VERSIONS[actual_version] + "^", False)

	print("Getting changed files...")
	def changed_files_setter(filename):
		current_commits_list = [commit for commit in current_commits if any(e for e in commit["files"].keys() if e.endswith(filename))]
		previous_commits_list = [commit for commit in previous_commits if any(e for e in commit["files"].keys() if e.endswith(filename))]
		return filename, {"current_commits": current_commits_list, "previous_commits": previous_commits_list}

	changed_files = {}
	with ThreadPoolExecutor() as executor:
		futures = [executor.submit(changed_files_setter, filename) for filename in filenames]
		for future in futures:
			filename, commits = future.result()
			changed_files[filename] = commits

	print("Calculating new metrics...")
	processed_dataset = _add_new_metrics(processed_dataset, changed_files, bugs)
	processed_dataset = processed_dataset.drop(columns=["Name"])
	return processed_dataset
