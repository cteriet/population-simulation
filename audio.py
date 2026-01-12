import numpy as np
import wave
import struct
import zlib
import os
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
from IPython.display import Audio, display

# --- CONFIGURATION ---------------------------------------------------------
class Config:
    SAMPLE_RATE = 44100
    FFT_SIZE = 4096
    CP_LEN = 512
    FREQ_MIN = 800
    FREQ_MAX = 12000
    SYNC_DURATION = 0.5
    SILENCE_GAP = 0.5
    REPETITIONS = 3
    MAGIC_SEQUENCE = b'\xAB\xCD\x12\x34'

    bin_width = SAMPLE_RATE / FFT_SIZE
    start_bin = int(FREQ_MIN / bin_width)
    end_bin = int(FREQ_MAX / bin_width)
    data_bins = np.arange(start_bin, end_bin)

# --- DIAGNOSTICS -----------------------------------------------------------
class Diagnostics:
    @staticmethod
    def plot_spectrogram(audio, title="Signal Spectrogram"):
        plt.figure(figsize=(10, 3))
        plt.specgram(audio + 1e-12, NFFT=1024, Fs=Config.SAMPLE_RATE, noverlap=512)
        plt.title(title)
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_constellation(diff_symbols, title="Differential Constellation (DQPSK)"):
        # Filter noise
        mag = np.abs(diff_symbols)
        active = diff_symbols[mag > 0.5]

        plt.figure(figsize=(5, 5))
        # Plot received
        plt.scatter(active.real, active.imag, s=5, alpha=0.6, label='Received')

        # Plot Ideals (0, 90, 180, 270 deg)
        ideals = [1+0j, 0+1j, -1+0j, 0-1j]
        plt.scatter(np.real(ideals), np.imag(ideals), c='red', marker='x', s=100, linewidth=3, label='Targets')

        plt.xlim(-2, 2); plt.ylim(-2, 2)
        plt.axhline(0, c='k', alpha=0.2); plt.axvline(0, c='k', alpha=0.2)
        plt.grid(True, linestyle=':')
        plt.legend(loc='lower right')
        plt.title(title)
        plt.tight_layout()
        plt.show()

# --- ENCODER ---------------------------------------------------------------
class OFDMEncoder:
    def __init__(self):
        self.cfg = Config()

    def generate_chirp(self):
        t = np.linspace(0, self.cfg.SYNC_DURATION, int(self.cfg.SAMPLE_RATE * self.cfg.SYNC_DURATION), endpoint=False)
        k = (8000 - 1000) / self.cfg.SYNC_DURATION
        return 0.5 * np.sin(2 * np.pi * (1000 * t + (k/2) * t**2))

    def encode_file(self, input_file, output_wav):
        print(f"\n[ENCODER] Encoding {input_file}...")
        with open(input_file, 'rb') as f: raw_data = f.read()

        # 1. Structure
        checksum = zlib.crc32(raw_data)
        header = self.cfg.MAGIC_SEQUENCE + struct.pack('>II', len(raw_data), checksum)
        payload = header + raw_data
        repeated_payload = payload * self.cfg.REPETITIONS

        # Pad with random bytes
        bits_per_symbol = 2
        bytes_per_ofdm_frame = (len(self.cfg.data_bins) * bits_per_symbol) // 8
        remainder = len(repeated_payload) % bytes_per_ofdm_frame
        if remainder != 0:
            repeated_payload += os.urandom(bytes_per_ofdm_frame - remainder)

        # 2. Map Bits to Phase Shifts
        bits = np.unpackbits(np.frombuffer(repeated_payload, dtype=np.uint8))
        bit_pairs = bits.reshape(-1, 2)

        phase_shifts = []
        for b0, b1 in bit_pairs:
            # DQPSK Mapping
            if b0==0 and b1==0: shift = 1.0 + 0.0j  # 0 deg
            elif b0==0 and b1==1: shift = 0.0 + 1.0j # 90 deg
            elif b0==1 and b1==0: shift = 0.0 - 1.0j # -90 deg
            else: shift = -1.0 + 0.0j # 180 deg
            phase_shifts.append(shift)

        phase_shifts = np.array(phase_shifts)

        # 3. Differential Encoding
        num_sub = len(self.cfg.data_bins)
        num_frames = len(phase_shifts) // num_sub
        grid_shifts = phase_shifts.reshape(num_frames, num_sub)

        # Initial State: Random Phases (White Noise-like)
        # This prevents the "Impulse" issue of the previous version
        np.random.seed(42) # Fixed seed not strictly needed for Rx, but good for debug
        current_phases = np.exp(1j * np.random.uniform(0, 2*np.pi, num_sub))

        frames_time_domain = []

        # Reference Frame
        fft_ref = np.zeros(self.cfg.FFT_SIZE, dtype=np.complex128)
        fft_ref[self.cfg.data_bins] = current_phases
        fft_ref[self.cfg.FFT_SIZE - self.cfg.data_bins] = np.conj(current_phases)
        t_ref = np.fft.ifft(fft_ref).real
        # Add CP and normalize
        sym_ref = np.concatenate([t_ref[-self.cfg.CP_LEN:], t_ref])
        sym_ref = sym_ref / (np.max(np.abs(sym_ref)) + 1e-12) * 0.5 # slightly quieter ref
        frames_time_domain.append(sym_ref)

        # Data Frames
        for i in range(num_frames):
            current_phases = current_phases * grid_shifts[i]
            current_phases /= np.abs(current_phases) # Keep mag 1.0

            fft_bins = np.zeros(self.cfg.FFT_SIZE, dtype=np.complex128)
            fft_bins[self.cfg.data_bins] = current_phases
            fft_bins[self.cfg.FFT_SIZE - self.cfg.data_bins] = np.conj(current_phases)

            t_sig = np.fft.ifft(fft_bins).real
            sym = np.concatenate([t_sig[-self.cfg.CP_LEN:], t_sig])
            sym = sym / (np.max(np.abs(sym)) + 1e-12) * 0.8 # Normal volume
            frames_time_domain.append(sym)

        # 4. Save
        chirp = self.generate_chirp()
        silence = np.zeros(int(self.cfg.SILENCE_GAP * self.cfg.SAMPLE_RATE))
        full_signal = np.concatenate([chirp, silence, np.concatenate(frames_time_domain), silence])

        display(Audio(data=full_signal, rate=self.cfg.SAMPLE_RATE, autoplay=True, normalize=False))

        scaled = (full_signal * 32767).astype(np.int16)
        with wave.open(output_wav, 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.cfg.SAMPLE_RATE)
            wf.writeframes(scaled.tobytes())

        print(f"[ENCODER] Created {output_wav}")
