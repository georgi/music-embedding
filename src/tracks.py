from typing import Optional, List, Dict, Any
import os
import sys
import numpy as np
import tqdm
import librosa
from concurrent.futures import ProcessPoolExecutor
import torch

N_FFT = 511
SEQ_LEN = 48
HOP_LENGTH = 256

class Track:
    file_path: str
    n_fft: int
    hop_length: int
    sr: int
    tempo: float
    signal: Optional[np.ndarray] = None

    def __init__(self, file_path: str, n_fft: int, hop_length: int, sr: int = 0, tempo: float = 0.0, segments: List[Dict[str, Any]] = []):
        self.file_path = file_path
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sr = sr
        self.tempo = tempo
        self.segments = [Segment(self, **s) for s in segments]

    def __repr__(self):
        return f"Track(file_path={self.file_path})"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'tempo': self.tempo,
            'segments': [s.to_dict() for s in self.segments],
            'file_path': self.file_path,
            'n_fft': self.n_fft,
            'hop_length': self.hop_length,
            'sr': self.sr
        }


    def save(self, cache_dir):
        f = os.path.join(cache_dir, os.path.basename(self.file_path))
        torch.save(self.to_dict(), f)


class Segment:
    track: Track
    start: int
    end: int
    seq_len: int
    spec: np.ndarray

    def __init__(self, track: Track, start: int, end: int, seq_len: int, spec: Optional[np.ndarray] = None):
        self.track = track
        self.start = start
        self.end = end
        self.seq_len = seq_len
        self.hop_length = (end - start) // seq_len
        self.spec = spec

    def load_signal(self, signal: np.ndarray, n_fft: int):
        stft = librosa.core.stft(
            signal[self.start:self.end],
            n_fft=n_fft,
            hop_length=self.hop_length
        )
        S, _ = librosa.core.magphase(stft)
        self.spec = np.log10(10000 * S + 1)
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            'start': self.start,
            'end': self.end,
            'seq_len': self.seq_len,
            'spec': self.spec
        }

    @property
    def signal(self) -> Optional[np.ndarray]:
        if self.track.signal:
            return self.track.signal[self.start:self.end]
        else:
            return None


def load_track(file_path: str, n_fft: int, hop_length: int, seq_len: int):
    track = Track(file_path, n_fft, hop_length)
    signal, sr = librosa.core.load(file_path)
    track.sr = sr
    spec: np.ndarray = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        hop_length=hop_length
    )
    spec = np.log10(10000 * spec + 1)
    onset_env = librosa.onset.onset_strength(
        S=spec,
        aggregate=np.median,
    )
    track.tempo = librosa.beat.tempo(
        onset_envelope=onset_env[:10000],
        sr=sr,
        hop_length=hop_length,
    )[0]
    _, onsets = librosa.beat.beat_track(
        bpm=track.tempo,
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        units='samples'
    )
    track.segments = [
        Segment(
            track=track,
            start=onsets[i],
            end=onsets[i + 1],
            seq_len=seq_len
        ).load_signal(signal, n_fft)
        for i in range(0, len(onsets) - 1)
    ]

    return track


def process_track(cache_dir: str, file_path: str):
    try:
        track = load_track(file_path, N_FFT, HOP_LENGTH, SEQ_LEN)
        track.save(cache_dir)
        return len(track.segments)
    except Exception:
        return 0


def process_tracks(cache_dir: str, files: List[str]):
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = []
        for file in files:
            future = executor.submit(process_track, cache_dir, file)
            futures.append(future)
        for future in tqdm.tqdm(futures):
            res = future.result()
            print(res)


if __name__ == '__main__':
    dataset_path = sys.argv[1]
    cache_dir = sys.argv[2]
    os.makedirs(cache_dir, exist_ok=True)
    files = []
    for dirpath, dirnames, filenames in os.walk(dataset_path):
        for f in filenames:
            file_path = os.path.join(dirpath, f)
            files.append(file_path)

    process_tracks(cache_dir, files)

