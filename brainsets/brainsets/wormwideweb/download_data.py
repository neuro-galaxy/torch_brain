import os
import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()
    output_dir = args.output_dir
    os.system("wget -P " + output_dir + " https://www.wormwideweb.org/readme_data.md")

    os.system(
        "wget -P " + output_dir + " https://www.wormwideweb.org/data/summary.json"
    )

    f = open(output_dir + "/summary.json")
    summary = json.load(f)

    keys = list(summary.keys())

    for key in keys:
        os.system(
            "wget -P "
            + output_dir
            + " https://www.wormwideweb.org/data/"
            + key
            + ".json"
        )


if __name__ == "__main__":
    main()
