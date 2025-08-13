import streamlit as st
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter, spectrogram
import matplotlib.pyplot as plt
import io

st.set_page_config(layout="wide")

def bandpass_filter(data, center_freq, bandwidth, fs, order=2):
    nyq = 0.5 * fs
    low = max(1e-5, (center_freq - bandwidth/2) / nyq)
    high = min(0.99999, (center_freq + bandwidth/2) / nyq)
    if low >= high:
        low = max(1e-5, high - 0.01)
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

def additive_sines_with_vibrato_and_envelope(partials, duration_ms, sample_rate):
    t = np.linspace(0, duration_ms / 1000, int(sample_rate * duration_ms / 1000), endpoint=False)
    wave = np.zeros_like(t)
    for p in partials:
        if p['vibrato'] and p['vib_depth'] > 0 and p['vib_rate'] > 0:
            instantaneous_freq = p['freq'] + p['vib_depth'] * np.sin(2 * np.pi * p['vib_rate'] * t)
            phase = 2 * np.pi * np.cumsum(instantaneous_freq) / sample_rate
        else:
            phase = 2 * np.pi * p['freq'] * t
        env = envelope(t, p['attack'], p['decay'], duration_ms)
        wave += p['amp'] * np.sin(phase) * env
    max_abs = np.max(np.abs(wave))
    if max_abs > 0:
        wave = wave / max_abs
    return wave

def envelope(t, attack_ms, decay_ms, total_ms):
    attack_samples = int((attack_ms / 1000) * len(t) / (total_ms / 1000))
    decay_samples = int((decay_ms / 1000) * len(t) / (total_ms / 1000))
    sustain_samples = len(t) - attack_samples - decay_samples
    env = np.ones_like(t)
    # Attack
    if attack_samples > 0:
        env[:attack_samples] = np.linspace(0, 1, attack_samples)
    # Decay
    if decay_samples > 0:
        env[-decay_samples:] = np.linspace(1, 0, decay_samples)
    # Sustain is already 1
    return env


st.title("Additive Vocal Synthesis with Formants")

# 1. Global controls
sample_rate = st.selectbox("Sample Rate (Hz)", [32000, 44100, 48000], index=1)
duration = st.slider("Total Duration (ms)", 100, 5000, 1000, step=100)

# 2. Partial (sine) controls
num_partials = st.slider("Number of Partials (Sines)", 1, 8, 3)
partial_cols = st.columns(num_partials)
partials = []
for i, col in enumerate(partial_cols):
    with col:
        st.markdown(f"**Partial {i+1}**")
        freq = st.slider(f"Frequency {i+1} (Hz)", 50, 2000, 220*(i+1), step=10, key=f"freq_{i}")
        amp = st.slider(f"Amplitude {i+1}", 0.05, 1.0, 0.5, step=0.05, key=f"amp_{i}")
        attack = st.slider(f"Attack {i+1} (ms)", 0, 1000, 50, step=10, key=f"attack_{i}")
        decay = st.slider(f"Decay {i+1} (ms)", 0, 1000, 200, step=10, key=f"decay_{i}")
        vibrato = st.checkbox(f"Vibrato", key=f"vib_{i}")
        if vibrato:
            vib_rate = st.slider(f"Vib Rate", 1, 20, 5, key=f"vib_rate_{i}")
            vib_depth = st.slider(f"Vib Depth (Hz)", 1, 200, 20, key=f"vib_depth_{i}")
        else:
            vib_rate = 0
            vib_depth = 0
        partials.append({
            'freq': freq,
            'amp': amp,
            'attack': attack,
            'decay': decay,
            'vibrato': vibrato,
            'vib_rate': vib_rate,
            'vib_depth': vib_depth
        })


# 3. Formant controls
num_formants = st.slider("Number of Formants", 1, 5, 2)
formant_freqs = []
formant_bandwidths = []
for i in range(num_formants):
    st.markdown(f"**Formant {i+1}**")
    f = st.slider(f"Formant {i+1} Frequency (Hz)", 200, 4000, 500 + 500*i, step=10, key=f"formant_freq_{i}")
    bw = st.slider(f"Formant {i+1} Bandwidth (Hz)", 50, 1000, 200, step=10, key=f"formant_bw_{i}")
    formant_freqs.append(f)
    formant_bandwidths.append(bw)

# 4. Synthesize
audio_data = additive_sines_with_vibrato_and_envelope(partials, duration, sample_rate)
for f, bw in zip(formant_freqs, formant_bandwidths):
    audio_data = bandpass_filter(audio_data, f, bw, sample_rate)

# 5. Normalize and output
max_abs = np.max(np.abs(audio_data))
if max_abs > 0:
    audio_data = audio_data / max_abs
audio_int16 = np.int16(audio_data * 32767)
total_duration = len(audio_int16) / sample_rate

# 6. Spectrogram and audio player
tick_interval = 0.25
xticks = np.arange(0, total_duration, tick_interval)
if not np.isclose(xticks[-1], total_duration):
    xticks = np.append(xticks, total_duration)

f, t_spec, Sxx = spectrogram(audio_int16.astype(float), fs=sample_rate, nperseg=1024)
dominant_freqs = f[np.argmax(Sxx, axis=0)]

fig, ax = plt.subplots(figsize=(14, 4))
pcm = ax.pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
fig.colorbar(pcm, ax=ax, label='dB')
ax.plot(t_spec, dominant_freqs, color='w', linewidth=1.5, label='Dominant Freq')
ax.set_ylabel('Frequency [Hz]')
ax.set_xlabel('Time [sec]')
ax.set_title('Spectrogram (with Dominant Frequency)')
ax.set_ylim(0, 5000)
ax.legend()
ax.set_xticks(xticks)
ax.set_xticklabels([f"{tick:.2f}" for tick in xticks])
st.pyplot(fig)

buf = io.BytesIO()
wavfile.write(buf, sample_rate, audio_int16)
st.audio(buf.getvalue(), format='audio/wav')

st.download_button(
    label="Save as WAV",
    data=buf.getvalue(),
    file_name="synthesized_vocal.wav",
    mime="audio/wav"
)
