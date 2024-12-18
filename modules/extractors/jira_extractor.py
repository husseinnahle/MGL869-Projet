import json
import requests

# Documentation de l'API de JIRA: https://developer.atlassian.com/cloud/jira/platform/rest/v2/intro
JIRA_URL = "https://issues.apache.org/jira/rest/api/2"
HEADERS = {"Accept": "application/json"}


def _extract_issues(filter: str, fields: list[str] = [], page: int = 0) -> list[dict]:
  """Extract JIRA issues.

  Args:
    filter (str): JQL filter.
    fields (list[str], optional): Field names to extract. Defaults to [].

  Returns:
    list[dict]: List of dictionnaries containning:
      - key (str): Jira key.
      - fields (dict[dict]): Jira fields.
  """
  try:
    params = {
      "jql": filter,
      "maxResults": 1000,
      "fieldsByKeys": True,
      "fields": fields,
      "startAt": page
    }
    response = requests.get(
      "https://issues.apache.org/jira/rest/api/2/search",
      headers={"Accept": "application/json"},
      params=params
    )
    response.raise_for_status()
    return [{"key": e["key"], "fields": e["fields"]} for e in json.loads(response.text)["issues"]]
  except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
    return []


def get_hive_issues(filter_by: list[str]) -> list[dict]:
  """Extraire billets Jira de bug du projet Hive.

  Returns:
      list[dict]: _description_
  """
  issues = {}
  page = 1
  filter = "project = HIVE AND issuetype = Bug AND status in (Resolved, Closed) AND resolution = Fixed"
  issues = _extract_issues(filter, filter_by, 1000*page)
  while len(issues) % 1000 == 0:
    page += 1
    issues += _extract_issues(filter, filter_by, 1000*page)
  return issues
