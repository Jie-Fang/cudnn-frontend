import argparse
import re
import os
import subprocess
import time

# should be in pytorch container
import requests

# matches
# <a*>x.y.w.z/</a> DD-month-YYYY HH:MM
pattern = re.compile(
    r"<a.*?>(\d+\.\d+\.\d+\.\d+)/</a>\s+(\d{2}-[A-Za-z]+-\d{4} \d{2}:\d{2})\s"
)


def download_url(url, path):
    # temporarily download to another location to avoid partial downloads
    temp_path = f"{path}.tmp"
    try:
        with requests.get(url, auth=("", os.getenv("JFROG_API_KEY")), stream=True) as r:
            r.raise_for_status()
            with open(temp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        os.rename(temp_path, path)
    except:
        raise
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def fetch_cudnn(base_url):
    response = requests.get(base_url).text
    matches = pattern.findall(response)
    matches = [{"version": a, "last_modified": b} for a, b in matches]

    # sort by version
    matches = sorted(
        matches, key=lambda x: tuple(map(int, x["version"].split("."))), reverse=True
    )

    top_three = matches[:3]

    for match in top_three:
        print(f"{match['version']} {match['last_modified']}")

    # download if not exists
    # if it fails to download, try a lower version one
    os.makedirs("downloads", exist_ok=True)
    for match in top_three:
        version = match["version"]
        path = f"downloads/cudnn-{version}.tar.gz"
        url = f"{base_url}/{version}/debug_cudnn-linux-x86_64-{version}.tar.gz"

        if os.path.exists(path):
            print(f"Fetch skipped for {version}: File already exists at {path}")
            break

        try:
            print(f"Fetching {version} from {url}")
            download_url(url, path)
            print(f"Fetching {version} complete")
            break
        except requests.exceptions.RequestException as e:
            print(f"WARNING: Fetching {version} from {url} failed: {e}")
        except Exception as e:
            raise Exception(f"ERROR: {e}")

    if not os.path.exists(path):
        raise Exception("ERROR: Failed to get any cuDNN build")

    # cleanup downloads older than 7 days
    current_time = time.time()
    for filename in os.listdir("downloads"):
        filepath = os.path.join("downloads", filename)
        if (
            os.path.isfile(filepath)
            and current_time - os.path.getmtime(filepath) > 7 * 24 * 60 * 60
        ):
            print(f"Cleaning up: {filepath}")
            os.remove(filepath)

    # extract
    print(f"Extracting {path}")
    subprocess.run(
        ["tar", "xvf", path, "--directory", "/"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    print(f"Extraction complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("base_url")
    args = parser.parse_args()
    fetch_cudnn(args.base_url)
