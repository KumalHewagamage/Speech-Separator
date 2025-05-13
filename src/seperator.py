import os
import torch
import librosa
import soundfile as sf  # modern audio writing

from utils.audio import Audio
from utils.hparams import HParam
from model.model import VoiceFilter
from model.embedder import SpeechEmbedder


class VoiceSeparator:
    def __init__(self, config_path, embedder_path, checkpoint_path, return_dvec=False):
        self.return_dvec = return_dvec
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load hyperparameters
        self.hp = HParam(config_path)

        # Load VoiceFilter model
        self.voice_filter = VoiceFilter(self.hp).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)["model"]
        self.voice_filter.load_state_dict(checkpoint)
        self.voice_filter.eval()

        # Load Speech Embedder
        self.embedder = SpeechEmbedder(self.hp).to(self.device)
        embed_ckpt = torch.load(embedder_path, map_location=self.device)
        self.embedder.load_state_dict(embed_ckpt)
        self.embedder.eval()

        # Initialize audio processor
        self.audio = Audio(self.hp)

    def separate(self, reference_file, mixed_file, out_dir=None):
        with torch.no_grad():
            # Extract d-vector from reference audio
            ref_wav, _ = librosa.load(reference_file, sr=16000)
            ref_mel = self.audio.get_mel(ref_wav)
            ref_mel = torch.from_numpy(ref_mel).float().to(self.device)
            dvec = self.embedder(ref_mel).unsqueeze(0)  # [1, emb_dim]
            # Prepare mixed audio
            mix_wav, _ = librosa.load(mixed_file, sr=16000)
            mag, phase = self.audio.wav2spec(mix_wav)
            mag = torch.from_numpy(mag).float().unsqueeze(0).to(self.device)

            # Predict mask and apply
            mask = self.voice_filter(mag, dvec)
            est_mag = (mag * mask)[0].cpu().numpy()
            est_wav = self.audio.spec2wav(est_mag, phase)

            # Optionally save to output directory
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, "result.wav")
                sf.write(out_path, est_wav, samplerate=16000)

            return (est_wav, dvec.cpu() if self.return_dvec else None)
