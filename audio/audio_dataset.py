import torchaudio
from torch.utils.data import Dataset
import torch


class AudioDataset(Dataset):
    """
    This class is responsible for creating the data loaders (information - paths etc) for each one of the audio files.
    """
    def __init__(self, file_paths: list, labels: list, max_length: int = 112000, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        label = self.labels[idx]
        # Load audio file
        waveform, sample_rate = torchaudio.load(audio_path)
        # Apply padding if necessary
        if waveform.size(1) < self.max_length:
            pad_amount = self.max_length - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        # Apply truncate if necessary
        elif waveform.size(1) > self.max_length:
            waveform = waveform[:, :self.max_length]

        label_tensor = torch.tensor(label)
        audio_path_name = audio_path.replace('\\', '/').split('/')[-1].split('.')[0]
        return audio_path_name, waveform, label_tensor
