"""
Coursera full catalog scraper — uses the public REST API (no auth required).
Fetches all ~19 000+ courses from api.coursera.org/api/courses.v1
"""

import csv
import time
import requests
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_URL   = "https://api.coursera.org/api/courses.v1"
PAGE_SIZE = 100   # max allowed by the API

FIELDS = ",".join([
    "name",
    "slug",
    "description",
    "partnerIds",
    "domainTypes",
    "primaryLanguages",
    "workload",
    "difficultyLevel",
    "certificates",
    "startDate",
    "courseType",
])

HEADERS = {
    "user-agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/145.0.0.0 Safari/537.36"
    )
}

CSV_FIELDS = [
    "id",
    "name",
    "slug",
    "url",
    "courseType",
    "description",
    "partnerIds",
    "domainIds",
    "subdomainIds",
    "primaryLanguages",
    "workload",
    "difficultyLevel",
    "certificates",
    "startDate",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_join(value) -> str:
    if isinstance(value, list):
        return "|".join(str(v) for v in value)
    return str(value) if value is not None else ""


def flatten(course: dict) -> dict:
    domain_types = course.get("domainTypes", [])
    domain_ids    = [d.get("domainId", "")    for d in domain_types]
    subdomain_ids = [d.get("subdomainId", "") for d in domain_types]
    slug = course.get("slug", "")
    return {
        "id":              course.get("id", ""),
        "name":            course.get("name", ""),
        "slug":            slug,
        "url":             f"https://www.coursera.org/learn/{slug}" if slug else "",
        "courseType":      course.get("courseType", ""),
        "description":     course.get("description", "").replace("\n", " ").replace("\r", ""),
        "partnerIds":      safe_join(course.get("partnerIds", [])),
        "domainIds":       safe_join(domain_ids),
        "subdomainIds":    safe_join(subdomain_ids),
        "primaryLanguages":safe_join(course.get("primaryLanguages", [])),
        "workload":        course.get("workload", ""),
        "difficultyLevel": course.get("difficultyLevel", ""),
        "certificates":    safe_join(course.get("certificates", [])),
        "startDate":       course.get("startDate", ""),
    }


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------

def scrape(output_path: str) -> None:
    session = requests.Session()
    session.headers.update(HEADERS)

    start = 0
    total_written = 0
    grand_total   = None

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()

        while True:
            params = {
                "start":  start,
                "limit":  PAGE_SIZE,
                "fields": FIELDS,
            }
            try:
                resp = session.get(API_URL, params=params, timeout=30)
                resp.raise_for_status()
            except requests.HTTPError as exc:
                print(f"\nHTTP error {exc.response.status_code} at start={start}. Stopping.")
                break
            except Exception as exc:
                print(f"\nError: {exc}. Stopping.")
                break

            data = resp.json()
            paging   = data.get("paging", {})
            elements = data.get("elements", [])

            if grand_total is None:
                grand_total = paging.get("total", "?")
                print(f"Total courses reported by API: {grand_total}\n")

            if not elements:
                print("No more elements. Done.")
                break

            for course in elements:
                writer.writerow(flatten(course))

            total_written += len(elements)
            next_start = paging.get("next")
            print(
                f"  start={start:6d}  got={len(elements):3d}  "
                f"total so far={total_written}/{grand_total}"
            )

            if not next_start:
                print("Reached last page. Done.")
                break

            start = int(next_start)
            time.sleep(0.25)

    print(f"\nSaved {total_written} rows to {output_path}")


if __name__ == "__main__":
    out = Path(__file__).parent.parent / "data" / "coursera.csv"
    scrape(str(out))
