"""
CLI entrypoint: run a single pipeline step (data prep, CNN, KD, or ensemble)
as defined in config.yaml.
"""
import yaml
import argparse

from src.data_prep import create_folds, balance_test_set, augment_train_set
from src.train_cnn import train_plain_cnn
from src.train_kd import train_kd
from src.ensemble_eval import run_ensemble


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main():
    parser = argparse.ArgumentParser(description="Stroke classification pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--step",
        type=str,
        required=True,
        choices=["data", "train_cnn", "train_kd", "ensemble"],
        help="Pipeline step: data, train_cnn, train_kd, or ensemble (run in order for a full experiment).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"=== Starting {config['project']['name']} ===")

    # Folder names for stratified split; order should match label indices in config.
    class_names = list(config["data"]["classes"].keys())

    if args.step == "data":
        print("\n--- [Step 1] Data preparation ---")
        base_path = config["data"]["base_path"]
        create_folds(
            base_path,
            base_path,
            n_splits=len(config["training"]["folds"]),
            classes=class_names,
        )
        balance_test_set(
            base_path, config["training"]["folds"], 750, classes=class_names
        )
        augment_train_set(
            base_path, config["training"]["folds"], 7500, classes=class_names
        )
        print("Data preparation steps finished.")

    elif args.step == "train_cnn":
        print("\n--- [Step 2] Baseline CNN training ---")
        train_plain_cnn(config)

    elif args.step == "train_kd":
        print("\n--- [Step 3] Knowledge distillation (KD) training ---")
        train_kd(config)

    elif args.step == "ensemble":
        print("\n--- [Step 4] Ensemble evaluation ---")

        ensemble_cfg = config.get("ensemble", {})
        sample_weights = ensemble_cfg.get("weights", {})
        validation_mode = ensemble_cfg.get("validation_mode", "internal")

        # Internal: fold held out under dataset/; external: separate root (e.g. Kaggle layout).
        if validation_mode == "internal":
            print("Mode: internal validation")
            data_path = ensemble_cfg.get(
                "internal_data_path", f"{config['data']['base_path']}/Fold1/test"
            )
            run_ensemble(
                config,
                data_path=data_path,
                model_weights=sample_weights,
                output_prefix="ensemble_internal",
            )
        else:
            print("Mode: external validation")
            data_path = ensemble_cfg.get("external_data_path", "")
            if not data_path:
                print("ERROR: external_data_path is not set in config.")
            else:
                run_ensemble(
                    config,
                    data_path=data_path,
                    model_weights=sample_weights,
                    output_prefix="ensemble_external",
                )

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
