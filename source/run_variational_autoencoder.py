import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import yaml  # type: ignore
from autoencoders import VariationalAutoEncoder, util
from autoencoders.dataloader import get_dataloader, get_default_shape
from torch.nn.functional import l1_loss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


def main() -> None:
    # Get command-line argument.
    parser = argparse.ArgumentParser(prog="AutoEncoder.py")
    parser.add_argument("mode", type=str, choices=["train", "test", "all"])
    parser.add_argument("config_file", type=str)
    args = parser.parse_args()

    params = util.read_params_from_yaml(args.config_file)

    invalid_channel = params["channel"] < 1
    invalid_width = params["width"] < 1
    invalid_height = params["height"] < 1
    if invalid_channel or invalid_width or invalid_height:
        params.update(get_default_shape(params["dataset_path"]))

    channel = params["channel"]
    image_size = (params["height"], params["width"])
    preprocess = setup_preprocess(channel, image_size)
    transform = transforms.Compose(preprocess)
    train_dataloader, test_dataloader, classes = get_dataloader(
        params["dataset_path"], transform, params
    )

    if args.mode == "train" or args.mode == "all":
        execute_training(params, train_dataloader)

    if args.mode == "test" or args.mode == "all":
        execute_testing(params, test_dataloader, classes)


def setup_preprocess(channel: int, image_size: Tuple[int, int]) -> List[Any]:
    if channel == 3:  # for RGB image
        preprocess = [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            torch.nn.Flatten(0),
        ]
    else:  # for grayscale image
        preprocess = [
            transforms.Grayscale(),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            torch.nn.Flatten(0),
        ]
    return preprocess


def execute_training(params: Dict[str, Any], dataloader: DataLoader) -> None:
    # Build autoencoder.
    device = params["device"]
    input_size = params["height"] * params["width"] * params["channel"]
    embedding_size = params["embedding_size"]
    latent_channel = params["latent_channel"]
    model = VariationalAutoEncoder(input_size, embedding_size, latent_channel)

    # Iterate training steps.
    loss_curve = list()
    max_epochs = params["max_epochs"]
    for _ in tqdm(range(max_epochs), desc="Training..."):
        train_loss = model.train_step(dataloader, device)
        loss_curve.append(train_loss)

    # Make output directory.
    output_dir = Path(params["output_dir_path"])
    output_dir.mkdir(exist_ok=True, parents=True)

    # Save model.
    filename = output_dir / params["model_filename"]
    model.save(filename)

    # Save loss_curve
    util.save_loss_curve(output_dir, loss_curve)

    filename = output_dir / "config.yaml"
    with open(filename, "w") as file:
        yaml.safe_dump(params, file, sort_keys=False)


def execute_testing(
    params: Dict[str, Any], dataloader: DataLoader, classes: List[str]
) -> None:
    output_dir = Path(params["output_dir_path"])

    # Load trained autoencoder.
    device = params["device"]
    image_shape = (params["channel"], params["height"], params["width"])
    filename = output_dir / params["model_filename"]
    model = VariationalAutoEncoder.load(filename)
    model.to(device)

    labels = list()
    embedding = list()
    loss = {"L1_loss": 0.0, "SSIM": 0.0, "PSNR": 0.0}

    # Evaluate model.
    counter = 0
    model.eval()
    with torch.no_grad():
        for x, label in tqdm(dataloader, desc="Testing..."):
            x = x.to(device)
            avg, var = model.encode(x)
            z = model.reparameterize(avg, var)
            y = model.decode(z)

            loss["L1_loss"] += l1_loss(x, y).item()
            loss["SSIM"] += util.ssim_loss(x, y, image_shape)
            loss["PSNR"] += util.psnr_loss(x, y, image_shape)

            labels.extend([classes[e] for e in label.tolist()])
            embedding.extend(avg.view(-1, model.latent_channel).tolist())

            if counter == 0:
                filename = output_dir / "reconstruction.png"
                org = x.view(
                    -1, image_shape[0], image_shape[1], image_shape[2]
                )
                rec = y.view(
                    -1, image_shape[0], image_shape[1], image_shape[2]
                )
                util.reconstruct_embeddings(filename, org, rec)
            counter += 1

    filename = output_dir / "embedding_vectors.png"
    util.plot_embeddings(filename, embedding, labels)

    filename = output_dir / "metrics.csv"
    num_batches = len(dataloader)
    for key in loss:
        loss[key] /= num_batches
        print(f"- {key:8}: {loss[key]}")
    util.dict_to_csv(filename, loss)


if __name__ == "__main__":
    main()
