import csv
import time
import requests

URL = "https://www.coursera.org/graphql-gateway?opname=Search"

HEADERS = {
    "accept": "application/json",
    "accept-language": "en",
    "apollographql-client-name": "search-v2",
    "content-type": "application/json",
    "operation-name": "Search",
    "origin": "https://www.coursera.org",
    "referer": "https://www.coursera.org/search?query=*",
    "user-agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/145.0.0.0 Safari/537.36"
    ),
    "dnt": "1",
}

QUERY = """
query Search($requests: [Search_Request!]!) {
  SearchResult {
    search(requests: $requests) {
      ...SearchResult
      __typename
    }
    __typename
  }
}

fragment SearchResult on Search_Result {
  elements {
    ...SearchHit
    __typename
  }
  facets {
    id
    values {
      id
      name
      __typename
    }
    __typename
  }
  pagination {
    cursor
    totalElements
    __typename
  }
  totalPages
  __typename
}

fragment SearchHit on Search_Hit {
  ...SearchProductHit
  __typename
}

fragment SearchProductHit on Search_ProductHit {
  id
  name
  url
  partners
  skills
  productType
  productDifficultyLevel
  productDuration
  avgProductRating
  numProductRatings
  isCourseFree
  isPartOfCourseraPlus
  isCreditEligible
  isNewContent
  tagline
  __typename
}
"""

CSV_FIELDS = [
    "id",
    "name",
    "url",
    "partners",
    "skills",
    "productType",
    "productDifficultyLevel",
    "productDuration",
    "avgProductRating",
    "numProductRatings",
    "isCourseFree",
    "isPartOfCourseraPlus",
    "isCreditEligible",
    "isNewContent",
    "tagline",
]

PAGE_SIZE = 24


def build_payload(cursor: str, facet_filters: list[dict] | None = None) -> list[dict]:
    return [
        {
            "operationName": "Search",
            "variables": {
                "requests": [
                    {
                        "entityType": "PRODUCTS",
                        "limit": PAGE_SIZE,
                        "facets": ["topic", "productDifficultyLevel", "productDuration",
                                   "productTypeDescription", "partners", "language",
                                   "isPartOfCourseraPlus"],
                        "sortBy": "BEST_MATCH",
                        "enableAutoAppliedFilters": False,
                        "requestOrigin": {"pageType": "SERP", "segmentType": "CONSUMER"},
                        "maxValuesPerFacet": 1000,
                        "facetFilters": facet_filters or [],
                        "cursor": cursor,
                        "query": "*",
                    }
                ]
            },
            "query": QUERY,
        }
    ]


def safe_join(value) -> str:
    if isinstance(value, list):
        return "|".join(str(v) for v in value)
    return str(value) if value is not None else ""


def parse_response(data: list | dict) -> tuple[list[dict], list[dict], str | None, int]:
    result_block = data[0] if isinstance(data, list) else data
    search_list = (
        result_block
        .get("data", {})
        .get("SearchResult", {})
        .get("search", [{}])
    )
    search = search_list[0] if search_list else {}

    elements = search.get("elements", [])
    facets = search.get("facets", [])
    pagination = search.get("pagination", {})
    total = pagination.get("totalElements", 0)
    next_cursor = pagination.get("cursor")

    products = [el for el in elements if el.get("__typename") == "Search_ProductHit"]
    return products, facets, next_cursor, total


def fetch(session: requests.Session, cursor: str, facet_filters=None):
    payload = build_payload(cursor, facet_filters)
    resp = session.post(URL, json=payload, timeout=30)
    resp.raise_for_status()
    return parse_response(resp.json())


def get_topics(session: requests.Session) -> list[dict]:
    """Return list of {id, name} topic facet values from an initial request."""
    _, facets, _, _ = fetch(session, "0")
    for facet in facets:
        if facet.get("id") == "topic":
            return facet.get("values", [])
    return []


def scrape_topic(session, topic_id: str, topic_name: str, seen: set, writer) -> int:
    facet_filters = [{"facetName": "topic", "multiSelectFacetValues": [topic_id]}]
    cursor = "0"
    written = 0

    while True:
        try:
            products, _, next_cursor, total = fetch(session, cursor, facet_filters)
        except requests.HTTPError as exc:
            print(f"  HTTP error: {exc}")
            break
        except Exception as exc:
            print(f"  Error: {exc}")
            break

        new = [p for p in products if p["id"] not in seen]
        for hit in new:
            seen.add(hit["id"])
            row = {field: safe_join(hit.get(field)) for field in CSV_FIELDS}
            if row["url"] and not row["url"].startswith("http"):
                row["url"] = "https://www.coursera.org" + row["url"]
            writer.writerow(row)
        written += len(new)

        if not products:
            break

        print(
            f"  [{topic_name}] cursor={cursor} → "
            f"{len(products)} products ({len(new)} new), total={total}"
        )

        if not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor
        time.sleep(0.4)

    return written


def scrape(output_path: str) -> None:
    session = requests.Session()
    session.headers.update(HEADERS)

    print("Fetching topic list...")
    topics = get_topics(session)
    print(f"Found {len(topics)} topics.\n")

    seen: set[str] = set()
    total_written = 0

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()

        for i, topic in enumerate(topics, 1):
            tid = topic.get("id", "")
            tname = topic.get("name", tid)
            print(f"[{i}/{len(topics)}] Topic: {tname}")
            n = scrape_topic(session, tid, tname, seen, writer)
            total_written += n
            print(f"  → {n} new rows (running total: {total_written})\n")
            time.sleep(0.3)

    print(f"Done. Saved {total_written} unique rows to {output_path}")


if __name__ == "__main__":
    import os
    out = os.path.join(os.path.dirname(__file__), "..", "data", "coursera.csv")
    out = os.path.normpath(out)
    scrape(out)
