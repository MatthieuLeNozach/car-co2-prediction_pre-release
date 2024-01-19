import os
import sys
sys.path.insert(0, '../src/')

import kaggle
import auto_co2 as co2


import requests

def download_file(url, filepath):
    response = requests.get(url)
    response.raise_for_status()  # Ensure we got a successful response

    with open(filepath, 'wb') as f:
        f.write(response.content)

    print(f"File has been downloaded to {filepath}.")


def main():
    download_file('https://github.com/user/repo/raw/branch/path/to/file', 'path/to/save/file')


if __name__ == '__main__':
    main()