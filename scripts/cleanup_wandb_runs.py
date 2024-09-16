import argparse
import logging
import os
import warnings
from pathlib import Path
from shutil import rmtree

import wandb

from scripts.cleanup_wandb_sweep import delete_artifacts

# python scripts/cleanup_wandb_runs.py --project MLP-CiFAR10 CNN-CiFAR10 ViT-CIFAR10

def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="Script to cleanup wandb runs from local disk and wandb. Provide run ids or tag the runs delete in the web client. If you do not wish to delete the run from wandb or local disk pass the correspondings arguments."
    )
    parser.add_argument(
        "--project", required=True, type=str, default=None, nargs="+", help="Wandb project"
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=os.environ.get("WANDB_ENTITY"),
        help="wandb entity",
    )
    parser.add_argument(
        "runs", type=str, help="Provide the run ids to clean up.", nargs="*"
    )

    parser.add_argument(
        "--cleanup-local-disk",
        action="store_false",
        default=True,
        help="If passed, the model_directories corresponding to the runs will not be deleted, default=True.",
    )
    parser.add_argument(
        "--delete-wandb-runs",
        action="store_true",
        default=False,
        help="If passed, the script will delete the wandb runs from the project, default=False",
    )
    parser.add_argument(
        "--delete-artifacts",
        action="store_false",
        default=True,
        help="If passed, the script will not delete the wandb artifacts from the project, default=True",
    )
    args = parser.parse_args()

    if args.entity is None:
        raise ValueError("A valid entity must be provided.")

    api = wandb.Api()
    print(args)
    for project in args.project:
        deleted_cnt = 0
        runs = api.runs(
            f"{args.entity}/{project}",
            filters={"$or": [{"id": {"$in": args.runs}}, {"tags": {"$in": ["delete", "delete-artifacts"]}}]},
        )
        logging.info(f"Processing {len(runs)} runs for project: {project}.")
        # import code; code.interact(local=locals()|globals())
        # exit(0)
        for run in runs:
            logging.info(f"\n\nProcessing run: {run.id}, ")
            if args.cleanup_local_disk:
                model_dir = run.config.get("model_dir", None)
                try:
                    if model_dir is not None and Path(model_dir).exists() and Path(model_dir).name != "experiments":
                        rmtree(model_dir)
                    else:
                        logging.info(f"Run {run.id} model dir ({model_dir} is not here still deleting")
                except PermissionError:
                    logging.info(f"Run {run.id} is logged from a different user, still deleting")
                    # continue
            if "delete-artifacts" in run.tags or "delete" in run.tags: # or args.delete_artifacts:
                # ans = input(f"Deleting {run.name} and all artifacts from wandb, continue? [y/n]")
                # if ans.lower() != "y":
                #     continue
                try:
                    delete_artifacts(run)
                except Exception as e:
                    print(f"Problem deleting artifacts {e}")
                    pass
                if "delete" in run.tags:# or args.delete_wandb_runs:
                    run.delete()
                    deleted_cnt += 1
        logging.info(f"Delete {deleted_cnt} runs from project {project}.")
            


if __name__ == "__main__":
    main()
