import os
import json
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from libs.transforms import get_transform
from libs.dataloader import SplitTableDataset
from libs.model import SplitModel
from libs.losses import split_loss

import time

from termcolor import cprint

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_dir",
        help="Path to training data.",
        required=True,
    )
    parser.add_argument(
        "--val_dir",
        help="Path to validation data.",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output_weight_path",
        dest="output_weight_path",
        help="Output folder path for model checkpoints and summary.",
        required=True,
    )
    parser.add_argument(
        "-e",
        "--num_epochs",
        type=int,
        dest="num_epochs",
        help="Number of epochs.",
        default=10,
    )
    # parser.add_argument(
    #     "-s",
    #     "--save_every",
    #     type=int,
    #     dest="save_every",
    #     help="Save checkpoints after given epochs",
    #     default=50,
    # )
    parser.add_argument(
        "--log_every",
        type=int,
        dest="log_every",
        help="Print logs after every given steps",
        default=10,
    )
    parser.add_argument(
        "--val_every",
        type=int,
        dest="val_every",
        help="perform validation after given steps",
        default=10,
    )
    parser.add_argument(
        "--lr",
        "--learning_rate",
        type=float,
        dest="learning_rate",
        help="learning rate",
        default=0.00075,
    )
    parser.add_argument(
        "--dr",
        "--decay_rate",
        type=float,
        dest="decay_rate",
        help="weight decay rate",
        default=0.5,
    )
    parser.add_argument(
        "--augment_tables",
        action="store_true",
        help="Apply augmentation on the tables"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue training from \"last_model.pth\" in output_weight_path."
    )
    parser.add_argument(
        "--load_model_from",
        help="Path to model file to fine-tune from."
    )

    configs = parser.parse_args()
    configs.__dict__['lr_step'] = 15

    print(25 * "=", "Configuration", 25 * "=")
    print("Train Directory:\t", configs.train_dir)
    print("Validation Directory:\t", configs.val_dir)
    print("Output Weights Path:\t", configs.output_weight_path)
    # print("Validation Split:\t", configs.validation_split)
    print("Number of Epochs:\t", configs.num_epochs)
    print("Continue:\t", configs.resume)
    print("Fine-tune from:\t", configs.load_model_from)
    # print("Save Checkpoint Frequency:", configs.save_every)
    print("Log after:\t", configs.log_every)
    print("Validate after:\t", configs.val_every)
    print("Batch Size:\t", 1)
    print("Learning Rate:\t", configs.learning_rate)
    print("Decay Rate:\t", configs.decay_rate)
    print("Augmentation:\t", configs.augment_tables)
    print(65 * "=")

    if configs.resume and configs.load_model_from:
        print("Error! Flags \"resume\" and \"load_model_from\" cannot both be set at the same time.")
        exit(0)

    batch_size = 1
    learning_rate = configs.learning_rate

    MODEL_STORE_PATH = configs.output_weight_path

    # train_images_path = configs.train_images_dir
    # train_labels_path = configs.train_labels_dir

    cprint("Loading dataset...", "blue", attrs=["bold"])
    train_dataset = SplitTableDataset(
        configs.train_dir,
        transforms=get_transform(train=True),
        fix_resize=False,
        augment=configs.augment_tables
    )
    val_dataset = SplitTableDataset(
        configs.val_dir,
        transforms=get_transform(train=True),
        fix_resize=False,
        augment=False
    )

    # split the dataset in train and test set
    torch.manual_seed(1)
    # indices = torch.randperm(len(dataset)).tolist()

    # test_split = int(configs.validation_split * len(indices))

    # train_dataset = torch.utils.data.Subset(dataset, indices[test_split:])
    # val_dataset = torch.utils.data.Subset(val_dataset, indices[:test_split])

    # define training and validation data loaders
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=1
    )
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    cprint("Creating split model...", "blue", attrs=["bold"])
    model = SplitModel().to(device)

    criterion = split_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=configs.lr_step, gamma=configs.decay_rate
    )

    if configs.resume and os.path.exists(MODEL_STORE_PATH):
        print("==============Resuming training from last checkpoint==============")
        checkpoint = torch.load(os.path.join(MODEL_STORE_PATH, "last_model.pth"))
        lr_scheduler.load_state_dict(checkpoint['scheduler'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
    else:
        os.makedirs(MODEL_STORE_PATH)

        with open(os.path.join(MODEL_STORE_PATH, "config.json"), 'w') as fp:
            json.dump(configs.__dict__, fp, sort_keys=True, indent=4)
        start_epoch = 0
        best_val_loss = 10000.

    if configs.load_model_from:
        print("=========Loading model from {}=========".format(configs.load_model_from))
        checkpoint = torch.load(configs.load_model_from)
        model.load_state_dict(checkpoint['model_state_dict'])

    num_epochs = configs.num_epochs

    # create the summary writer
    writer = SummaryWriter(os.path.join(MODEL_STORE_PATH, "summary"))

    # Train the model
    total_step = len(train_loader)

    print(27 * "=", "Training", 27 * "=")

    step = 0

    val_iter = iter(val_loader)

    try:
        val_batch = next(val_iter)
    except StopIteration:
        val_batch = None

    time_stamp = time.time()
    for epoch in range(start_epoch, num_epochs):

        val_loss_list = []
        for i, (images, targets, img_path, _, _) in enumerate(train_loader):
            images = images.to(device)

            model.train()
            # incrementing step
            step -= -1

            targets[0] = targets[0].long().to(device)
            targets[1] = targets[1].long().to(device)

            # Backprop and perform Adam optimisation
            optimizer.zero_grad()

            # Run the forward pass
            outputs = model(images.to(device))
            loss, rpn_loss, cpn_loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            if (i + 1) % configs.log_every == 0:
                # writing loss to tensorboard
                writer.add_scalar(
                    "total loss train", loss.item(), (epoch * total_step + i)
                )
                writer.add_scalar(
                    "rpn loss train", rpn_loss.item(), (epoch * total_step + i)
                )
                writer.add_scalar(
                    "cpn loss train", cpn_loss.item(), (epoch * total_step + i)
                )
                # cprint("Iteration: ", "green", attrs=["bold"], end="")
                # print(step)
                # cprint("Learning Rate: ", "green", attrs=["bold"], end="")
                # print(lr_scheduler.get_last_lr()[0])
                print(
                    "Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, RPN Loss: {:.4f}, CPN Loss: {:.4f}, Learning Rate: {:.6f}, Time taken: {:.2f}s".format(
                        epoch + 1,
                        num_epochs,
                        i + 1,
                        total_step,
                        loss.item(),
                        rpn_loss.item(),
                        cpn_loss.item(),
                        lr_scheduler.get_last_lr()[0],
                        time.time() - time_stamp
                    )
                )
                time_stamp = time.time()

            if (i + 1) % configs.val_every == 0:
                model.eval()
                with torch.no_grad():
                    val_images, val_targets, _, _, _ = val_batch

                    val_targets[0] = val_targets[0].long().to(device)
                    val_targets[1] = val_targets[1].long().to(device)

                    val_outputs = model(val_images.to(device))
                    val_loss, val_rpn_loss, val_cpn_loss = criterion(
                        val_outputs, val_targets
                    )

                    val_loss_list.append(val_loss.item())
                    print(
                        "Validation iteration: Val Loss: {:.4f}, RPN Loss: {:.4f}, CPN Loss: {:.4f}".format(
                            val_loss,
                            val_rpn_loss,
                            val_cpn_loss,
                        )
                    )
                    print("---")

                    val_loss_list.append(val_loss.item())
                    writer.add_scalar(
                        "total loss val", val_loss.item(), (epoch * total_step + i)
                    )
                    writer.add_scalar(
                        "rpn loss val", val_rpn_loss.item(), (epoch * total_step + i)
                    )
                    writer.add_scalar(
                        "cpn loss val", val_cpn_loss.item(), (epoch * total_step + i)
                    )

                    try:
                        val_batch = next(val_iter)
                    except StopIteration:
                        val_iter = iter(val_loader)
                        val_batch = next(val_iter)

            # if (step + 1) % configs.save_every == 0:
        print(65 * "=")
        print("Saving model weights at epoch", epoch + 1)

        lr_scheduler.step()

        torch.save(
            {
                "epoch": epoch + 1,
                "best_val_loss": best_val_loss,
                "scheduler": lr_scheduler.state_dict(),
                # "iteration": step + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                # "config": configs
            },
            os.path.join(MODEL_STORE_PATH, "last_model.pth"),
        )

        if best_val_loss > sum(val_loss_list) / len(val_loss_list):
            with open(os.path.join(MODEL_STORE_PATH, "best_epoch.txt"), 'w') as f:
                f.write(str(epoch))
            best_val_loss = sum(val_loss_list) / len(val_loss_list)   
            torch.save(
                {
                    "epoch": epoch + 1,
                    # "iteration": step + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    # "config": configs
                },
                os.path.join(MODEL_STORE_PATH, "best_model.pth"),
            )   

        print(65 * "=")

        torch.cuda.empty_cache()


