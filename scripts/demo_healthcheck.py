"""Check the served model identity, real client evidence, and agent path."""

import argparse
import json
from urllib.request import Request, urlopen

from transaction_forecasting.product.provenance import model_lock


def check(base_url: str) -> dict:
    def request(path, method="GET"):
        with urlopen(Request(base_url + path, method=method), timeout=60) as response:
            return json.load(response)

    health = request("/api/v1/health")
    if health["status"] != "ready" or health["base_sha"] != model_lock()["base_sha"]:
        raise RuntimeError("Healthcheck returned a different frozen model")
    cases = request("/api/v1/cases")
    for case in cases:
        path = f"/api/v1/clients/{case['client_id']}"
        client = request(path)
        agent = request(path + "/investigate", "POST")
        if agent["predicted_family"] != client["prediction"]["predicted_family"]:
            raise RuntimeError("Agent changed the prediction")
        if not client["history"] or not agent["trace"]:
            raise RuntimeError("Missing real history or investigation")
    return {"health": health, "cases_checked": len(cases)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    print(json.dumps(check(parser.parse_args().url)))
