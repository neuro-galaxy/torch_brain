import os
import argparse

from allensdk.core.brain_observatory_cache import BrainObservatoryCache


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--session_id", type=str)
    parser.add_argument(
        "--output_dir", type=str, default="./raw", help="Output directory"
    )
    args = parser.parse_args()

    manifest_path = os.path.join(args.output_dir, "manifest.json")
    boc = BrainObservatoryCache(manifest_file=manifest_path)

    truncated_file = True
    file_path = os.path.join(
        args.output_dir, f"ophys_experiment_data/{args.session_id}.nwb"
    )

    while truncated_file:
        try:
            exp = boc.get_ophys_experiment_data(
                file_name=file_path, ophys_experiment_id=int(args.session_id)
            )
            truncated_file = False
        except OSError:
            os.remove(
                os.path.join(
                    args.output_dir, f"ophys_experiment_data/{args.session_id}.nwb"
                )
            )
            print(" Truncated file, re-downloading")
