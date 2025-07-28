import base64
import io
import modal
import torch
import librosa
import requests
import torch.nn as nn
import torchaudio.transforms as T
import numpy as np
import soundfile as sf
from pydantic import BaseModel
from model import AudioCNN

app = modal.App("AudioCNN-Inference")

image = (modal.Image.debian_slim()
         .pip_install_from_requirements("requirements.txt")
         .apt_install(["libsndfile1"])
         .add_local_python_source("model")
         )

# already created in training
model_volume = modal.Volume.from_name("esc-model")


class AudioProcessor:
    def __init__(self):
        self.transform = nn.Sequential(
            # standard MelSpectrogram configuration values
            T.MelSpectrogram(
                sample_rate=44100,
                n_fft=1024,  # windows size
                hop_length=512,
                n_mels=128,
                f_min=0,
                f_max=11025
            ),
            T.AmplitudeToDB(),
        )

    def process_audio_chunk(self, audio_data):
        waveform = torch.from_numpy(audio_data)
        # add a channel to waveform tensor
        waveform = waveform.unsqueeze(0)
        spectrogram = self.transform(waveform)
        # add model dimension - model expected format
        return spectrogram.unsqueeze(0)


class InferenceRequest(BaseModel):
    audio_data: str


@app.cls(image=image, gpu="A10G", volumes={"/models": model_volume}, scaledown_window=15)
class AudioClassfier:
    @modal.enter()
    def load_model(self):
        print("Loading model on enter.")
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load("/models/best_model.pth",
                                map_location=self.device)
        self.classes = checkpoint["classes"]
        self.model = AudioCNN(num_classes=len(self.classes))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.audio_processor = AudioProcessor()
        print("Model loaded successfully on enter.")

    @modal.fastapi_endpoint(method="POST")
    def inference(self, request: InferenceRequest):
        # decode base64 audio from payload
        # production: frontend -> upload file to S3 -> inference endpoint -> download file from S3
        # in this project: frontend -> send file directly to inference endpoint
        audio_bytes = base64.b64decode(request.audio_data)
        audio_data, sample_rate = sf.read(
            io.BytesIO(audio_bytes), dtype="float32")

        # keep audio in 1 channel (Mono) instead of multiple channels
        # if more than 1 channel, take the mean of channels
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        # keep sample rate the same as our configuration
        if sample_rate != sample_rate:
            audio_data = librosa.resample(
                y=audio_data, orig_sr=sample_rate, target_sr=sample_rate)

        spectogram = self.audio_processor.process_audio_chunk(audio_data)
        # move spectrogram to device
        spectogram = spectogram.to(self.device)

        with torch.no_grad():
            output, features_map = self.model(
                spectogram, return_features_map=True)
            # handles nan just in case model produces nan for any reason
            output = torch.nan_to_num(output)
            # dim=0 batch, dim=1 class (batch_size, num_class)
            probabilities = torch.softmax(output, dim=1)
            # [0] - we only have single item in our batch
            top3_probs, top3_indices = torch.topk(probabilities[0], 3)
            # ex: top3_probs = [0.9, 0.01, 0.08], top3_indices = [15, 42, 5]
            # zip: [(0.9, 15), (0.01, 42), (0.08, 5)]
            predictions = [{"class": self.classes[idx.item()], "confidence": prob.item()}
                           for prob, idx in zip(top3_probs, top3_indices)]
            viz_data = {}
            for name, tensors in features_map.items():
                if tensors.dim() == 4:  # [batch_size, channels, height, width]
                    # 1 = channels. Take the mean of all channels
                    # this api calls remove channels from tensor [batch_size, height, width]
                    aggregated_tensor = torch.mean(tensors, dim=1)
                    # remove batch_size [height, width]
                    squeezed_tensor = aggregated_tensor.squeeze(dim=0)
                    # move tensor to cpu and convert to numpy array
                    numpy_arr = squeezed_tensor.cpu().numpy()
                    # convert NaN to 0
                    clean_arr = np.nan_to_num(numpy_arr)
                    viz_data[name] = {
                        "shape": list(clean_arr.shape),
                        "clean_array": clean_arr.tolist()
                    }
            # [batch_size, channels, height, width]
            # remove batch_size and channels - [height, width]
            spectrogram_numpy = spectogram.squeeze(0).squeeze(0).cpu().numpy()
            clean_spectogram = np.nan_to_num(spectrogram_numpy)

            # grab waveform but max 8000 samples
            max_samples = 8000
            if len(audio_data) > max_samples:
                # ex: 100,000 / 8000 = 12.5 -> 13
                step = len(audio_data) // max_samples
                # create a new array - from start to finish but will take every step/n element
                waveform_data = audio_data[::step]
            else:
                waveform_data = audio_data

        response = {
            "predictions": predictions,
            "visualizations": viz_data,
            "input_spectrogram": {
                "shape": list(clean_spectogram.shape),
                "clean_array": clean_spectogram.tolist()
            },
            "waveform": {
                "values": waveform_data.tolist(),
                "sample_rate": sample_rate,
                "duration": len(audio_data) / sample_rate
            }

        }
        return response


@app.local_entrypoint()
def main():
    # mirror frontend call
    audio_data, _ = sf.read("vacuum_cleaner.wav")
    buffer = io.BytesIO(audio_data)
    # load audio to memory
    sf.write(buffer, audio_data, 44100, format="wav")
    audio_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    payload = {"audio_data": audio_b64}
    server = AudioClassfier()
    url = server.inference.get_web_url()
    response = requests.post(url, json=payload)
    response.raise_for_status()
    result = response.json()
    waveform_info = result.get("waveform", {})
    if waveform_info:
        values = waveform_info.get("values", [])
        print(
            f"First 10 Waveform Values: {[round(v, 4) for v in values[:10]]}...")
        print(f"Waveform Duration: {waveform_info.get('duration', 0)}")

    print("Top Predictions:")
    for prediction in result.get("predictions"):
        print(f"{prediction['class']}: {prediction['confidence']:0.2%}")
