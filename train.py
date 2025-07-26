import modal
import pandas as pd
import torch
from torch.optim.lr_scheduler import OneCycleLR
import torchaudio
import torch.nn as nn
import torch.optim as optim
import torchaudio.transforms as T
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
from model import AudioCNN

app = modal.App("AudioCNN")

artifact_image = (modal.Image.debian_slim()
                  .pip_install_from_requirements("requirements.txt")
                  .apt_install(["wget", "unzip", "ffmpeg", "libsndfile1"])
                  .run_commands([
                      "cd /tmp && wget https://github.com/karolpiczak/ESC-50/archive/master.zip -O esc50.zip",
                      "cd /tmp && unzip esc50.zip",
                      "mkdir -p /opt/esc50-data",
                      "cp -r /tmp/ESC-50-master/* /opt/esc50-data/",
                      "rm -rf /tmp/esc50.zip /tmp/ESC-50-master"
                  ])
                  .add_local_python_source("model")
                  )

# create a volume to persist downloaded data in the docker container
# this volume attaches to the downloaded ESC-50 dataset that was copied to /opt/esc50-data
# by using from_name("esc50-data"), we're either creating a new volume or reusing an existing one
# create_if_missing=True ensures the volume exists, creating it if needed
volume = modal.Volume.from_name("esc50-data", create_if_missing=True)
# this volume attaches to the model which we will generate whenever we train the model
# model files from training will stored in this volume
# when we run the inference, we don't need the data volume
# this is the reason why two volumes are used
model_volume = modal.Volume.from_name("esc-model", create_if_missing=True)


class ES50Dataset(Dataset):
    # load in memory
    # In ES50 dataset, there are audio and metadata folders
    def __init__(self, data_dir, metadata_file, split="train", transform=None):
        super().__init__()
        self.data_dir = Path(data_dir)
        # metadata file is a csv - we need Pandas
        self.metadata = pd.read_csv(metadata_file)
        self.split = split
        self.transform = transform

        # fold column contains 5 different values
        # each value is divide into 400 rows - total 2000 rows
        # removing 5 remains 80% of the training data
        if split == "train":
            self.metadata = self.metadata[self.metadata["fold"] != 5]
        else:
            self.metadata = self.metadata[self.metadata["fold"] == 5]

        # 50 classes/catagories
        self.classes = sorted(self.metadata["category"].unique())
        # class to integer mapping to work with computer
        self.classes_to_idxs = {cls: idx for idx,
                                cls in enumerate(self.classes)}
        self.metadata["label"] = self.metadata["category"].map(
            self.classes_to_idxs)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        audio_path = self.data_dir / "audio" / row["filename"]
        waveform, sample_rate = torchaudio.load(audio_path)
        # waveform example: [channels, samples] = [2, 44000] -> [1, 44000]
        # (taking the mean of these two channels)
        if waveform.shape[0] > 1:
            # dim=channel dimension
            # keep the dimension
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if self.transform:
            spectrogram = self.transform(waveform)
        else:
            spectrogram = waveform

        return spectrogram, row["label"]


def mixup_data(features, labels):
    # create a beta distribution for both x and y
    # blending percentage
    lam = np.random.beta(0.2, 0.2)
    batch_size = features.size(0)
    # shuffle the batch of audio clip
    index = torch.randperm(batch_size).to(features.device)
    # linear interpolation
    # ex: (0.7 * audio1) + (0.3 * audio2)
    # first audio clip + second audio clip
    mixed_features = lam * features + (1 - lam) * features[index, :]
    label_a, label_b = labels, labels[index]
    return mixed_features, label_a, label_b, lam


def mixedup_criterion(criterion, prediction, label_audio1, label_audio2, lam):
    # percentage * calculate loss
    # calculate the loss as 100% original sound
    return lam * criterion(prediction, label_audio1) + (1-lam) * criterion(prediction, label_audio2)
# pass in required image, gpu, volumes and timeout in order to run training


@app.function(
    image=artifact_image,
    gpu="A10G",
    volumes={
        "/data": volume,
        "/models": model_volume
    },
    timeout=60*60*3)
def train():
    from datetime import datetime
    timestampe = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"/models/tensorboard_logs/run_{timestampe}"
    writer = SummaryWriter(log_dir)
    esc50_dir = Path("/opt/esc50-data")
    train_transform = nn.Sequential(
        # standard MelSpectrogram configuration values
        T.MelSpectrogram(
            sample_rate=22050,
            n_fft=1024,  # windows size
            hop_length=512,
            n_mels=128,
            f_min=0,
            f_max=11025
        ),
        T.AmplitudeToDB(),
        # Masking is like dropout - we don't need to mask in validation
        T.FrequencyMasking(freq_mask_param=30),
        T.TimeMasking(time_mask_param=80)
    )

    validate_transform = nn.Sequential(
        # standard MelSpectrogram configuration values
        T.MelSpectrogram(
            sample_rate=22050,
            n_fft=1024,  # windows size
            hop_length=512,
            n_mels=128,
            f_min=0,
            f_max=11025
        ),
        T.AmplitudeToDB()
    )

    train_data = ES50Dataset(
        data_dir=esc50_dir,
        metadata_file=esc50_dir / "meta" / "esc50.csv",
        split="train",
        transform=train_transform
    )
    validate_data = ES50Dataset(
        data_dir=esc50_dir,
        metadata_file=esc50_dir / "meta" / "esc50.csv",
        split="validation",
        transform=validate_transform
    )

    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=32,
        shuffle=True
    )
    # shuffle=False because it will retains the original data order
    validate_dataloader = DataLoader(
        dataset=validate_data,
        batch_size=32,
        shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioCNN(num_classes=len(train_data.classes))
    # move initialised model to device
    model.to(device)
    num_epochs = 100
    # Loss Function
    # label smoothing forces model to be humble in prediction
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(), lr=0.0005, weight_decay=0.001)
    # scheduler adjust learning rate during training so that model can learn more effectively
    # scheduler needs access to optimizer so that it can adjust lr
    scheduler = OneCycleLR(
        optimizer=optimizer,
        max_lr=0.002,
        epochs=num_epochs,
        steps_per_epoch=len(train_dataloader),
        pct_start=0.1
    )

    best_accuracy = 0.0
    print("Starting Training")
    for epoch in range(num_epochs):
        # set model to training mode
        model.train()
        # accumulate loss batch by batch
        # add up all batches losses
        epoch_loss = 0.0
        progress_bar = tqdm(
            train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        # data=MelSpectrogram, target=label
        for data, target in progress_bar:
            data, target = data.to(device), target.to(device)
            # technique for making model smart - data mixing
            # mixing together sounds - ex: dog merge with car horn sound
            # train the model on merged sound
            # benefits: model needs to work harder at extracting dog sound
            # meaning better at detecting dog bark in the future
            # real world sound isn't as isolated as in the dataset
            # prevent model from over confident - instead of saying it's 100% sure
            # more reliable in the future
            # mix up only 30% of the time
            if np.random.random() > 0.7:
                data, target_a, target_b, lam = mixup_data(data, target)
                output = model(data)
                # calculate loss differently for data mixed up
                loss = mixedup_criterion(
                    criterion, output, target_a, target_b, lam)
            else:
                output = model(data)
                loss = criterion(output, target)

            # reset the gradient
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = epoch_loss / len(train_dataloader)
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar(
            "Learning Rate", optimizer.param_groups[0]["lr"], epoch)
        # Validation after each epoch
        model.eval()
        correct = 0
        total = 0
        validation_loss = 0
        # don't change weights and biases
        # not training, testing
        with torch.no_grad():
            for data, target in validate_dataloader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                validation_loss += criterion(outputs, target).item()

                _, predicted = torch.max(outputs.data, 1)
                # grab the batch size and add it to the total
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        avg_validation_loss = validation_loss / len(validate_dataloader)
        print(
            f"Epoch {epoch + 1}. Loss: {avg_train_loss:.4f}, Validation Loss: {avg_validation_loss:.4f}, Accuracy: {accuracy:.2f}%")
        writer.add_scalar("Loss/Validation", avg_validation_loss, epoch)
        writer.add_scalar("Accuracy/Validation", accuracy, epoch)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            # save model checkpoint
            torch.save({
                "model_state_dict": model.state_dict(),
                "accuracy": accuracy,
                "epoch": epoch,
                "classes": train_data.classes
            }, "/models/best_model.pth")
            print(f"New best model saved: {accuracy:.2f}%")

    writer.close()
    print(f"Training completed. Best accuracy: {best_accuracy:.2f}%")


@app.local_entrypoint()
def main():
    print(train.remote())
