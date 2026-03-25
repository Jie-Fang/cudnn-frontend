import argparse
import re
import os
import subprocess
import time
from pathlib import Path

# should be in pytorch container
import requests

JFROG_API_KEY = os.getenv("JFROG_API_KEY")

# matches
# <a*>x.y.w.z/</a> DD-month-YYYY HH:MM
PATTERN = re.compile(r"<a.*?>v(\d+\.\d+\.\d+\.\d+)/</a>\s+(\d{2}-[A-Za-z]+-\d{4} \d{2}:\d{2})\s")


def download_url(url, path):
    # temporarily download to another location to avoid partial downloads
    temp_path = Path(f"{path}.tmp")
    try:
        with requests.get(url, auth=("", JFROG_API_KEY) if JFROG_API_KEY else None, stream=True) as r:
            r.raise_for_status()
            with temp_path.open("wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        temp_path.rename(path)
    except:
        raise
    finally:
        if temp_path.exists():
            temp_path.unlink()


def fetch_cudnn(base_url, cuda_version, download_dir, unzip_dir, output_dir, cudnn_version=None):
    response = requests.get(base_url, auth=("", JFROG_API_KEY) if JFROG_API_KEY else None).text
    matches = PATTERN.findall(response)
    matches = [{"version": a, "last_modified": b} for a, b in matches]

    # sort by version and filter out the requested version if provided
    matches = sorted(matches, key=lambda x: tuple(map(int, x["version"].split("."))), reverse=True)
    candidates = [{"version": cudnn_version, "last_modified": "requested"}] if cudnn_version is not None else matches[:3]
    for match in candidates:
        print(f"{match['version']} {match['last_modified']}")

    # download if not exists
    # if it fails to download, try a lower version one
    downloads_dir = Path(download_dir)
    downloads_dir.mkdir(exist_ok=True)
    tarball_path = None
    for match in candidates:
        version = match["version"]
        tarball_path = downloads_dir / f"cudnn-{version}.tar.gz"
        url = f"{base_url}/v{version}/{cuda_version}/cudnn_debug-linux-x86_64-{version}.tar.gz"

        if tarball_path.exists():
            print(f"Fetch skipped for {version}: File already exists at {tarball_path}")
            break

        try:
            print(f"Fetching {version} from {url}")
            download_url(url, tarball_path)
            print(f"Fetching {version} complete")
            break
        except requests.exceptions.RequestException as e:
            print(f"WARNING: Fetching {version} from {url} failed: {e}")
        except Exception as e:
            raise Exception(f"ERROR: {e}")

    if tarball_path is None or not tarball_path.exists():
        raise Exception("ERROR: Failed to get any cuDNN build")

    # extract, move, and copy
    print(f"Extracting {tarball_path}")
    subprocess.run(["tar", "xvf", str(tarball_path), "--directory", unzip_dir], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    print(f"Moving {Path(unzip_dir) / 'cudnn'} to {output_dir}")
    subprocess.run(["mv", str(Path(unzip_dir) / "cudnn"), output_dir], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    print(f"Copying {Path(output_dir) / 'lib'} to {Path(output_dir) / 'lib64'}")
    subprocess.run(["cp", "-r", str(Path(output_dir) / "lib"), str(Path(output_dir) / "lib64")], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    print(f"fetch_cudnn complete")

    # cleanup downloads older than 7 days
    try:
        current_time = time.time()
        for filepath in downloads_dir.iterdir():
            if filepath.is_file() and current_time - filepath.stat().st_mtime > 7 * 24 * 60 * 60:
                print(f"Cleaning up: {filepath}")
                filepath.unlink()
    except Exception as e:
        print(f"WARNING: Cleanup failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch a cuDNN debug tarball from URM/JFrog, extract it to /debug_cudnn, and clean old cached downloads.")
    parser.add_argument("--base-url", dest="base_url", required=True, help="Base repository URL containing versioned cuDNN folders, e.g. https://urm.nvidia.com/artifactory/hw-cudnn-generic/CUDNN/v9.21")
    parser.add_argument("--cuda-version", dest="cuda_version", required=True, help="CUDA version subdirectory to fetch from, e.g. 13.2")
    parser.add_argument("--download-dir", dest="download_dir", default="downloads", help="Directory where downloaded tarballs are cached.")
    parser.add_argument("--unzip-dir", dest="unzip_dir", default="/", help="Directory where the tarball is extracted before moving cudnn/ into place.")
    parser.add_argument("--output-dir", dest="output_dir", default="/debug_cudnn", help="Directory where the extracted cudnn/ tree will be moved.")
    parser.add_argument("--cudnn-version", dest="cudnn_version", help="Optional cuDNN version in x.y.w.z format, e.g. --cudnn-version 9.21.0.3")
    args = parser.parse_args()
    fetch_cudnn(args.base_url, args.cuda_version, args.download_dir, args.unzip_dir, args.output_dir, args.cudnn_version)
