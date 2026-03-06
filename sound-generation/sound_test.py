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

def generate_waveform_with_harmonics(
    waveform, f_start, f_end, amplitude, duration_ms, sample_rate, num_harmonics=5,
    vibrato=False, vib_rate=0, vib_depth=0
):
    t = np.linspace(0, duration_ms / 1000, int(sample_rate * duration_ms / 1000), endpoint=False)
    if f_start == f_end:
        base_freq = f_start
        if vibrato and vib_depth > 0 and vib_rate > 0:
            instantaneous_freq = base_freq + vib_depth * np.sin(2 * np.pi * vib_rate * t)
            phase = 2 * np.pi * np.cumsum(instantaneous_freq) / sample_rate
        else:
            phase = 2 * np.pi * base_freq * t
    else:
        # Frequency sweep (linear) with optional vibrato
        T = duration_ms / 1000
        base_freq = f_start + (f_end - f_start) * t / T
        if vibrato and vib_depth > 0 and vib_rate > 0:
            instantaneous_freq = base_freq + vib_depth * np.sin(2 * np.pi * vib_rate * t)
            phase = 2 * np.pi * np.cumsum(instantaneous_freq) / sample_rate
        else:
            phase = 2 * np.pi * (f_start * t + 0.5 * (f_end - f_start) * t**2 / T)

    wave = np.zeros_like(t)
    for n in range(1, num_harmonics+1):
        if waveform == 'sine':
            partial = np.sin(n * phase)
        elif waveform == 'square':
            partial = np.sign(np.sin(n * phase))
        elif waveform == 'triangle':
            partial = 2 * np.abs(2 * ((n * phase) % (2 * np.pi)) / (2 * np.pi) - 1) - 1
        elif waveform == 'sawtooth':
            partial = 2 * ((n * phase) % (2 * np.pi)) / (2 * np.pi) - 1
        else:
            raise ValueError("Unsupported waveform")
        wave += (amplitude / n) * partial
    max_abs = np.max(np.abs(wave))
    if max_abs > 0:
        wave = wave / max_abs
    return wave



st.title("Configurable Sound Synthesizer with Live Spectrogram")

# 1. Global controls
sample_rate = st.selectbox("Sample Rate (Hz)", [32000, 44100, 48000], index=1)
num_call = st.slider("Number of Calls", 1, 6, 2)
intercall_interval_ms = st.slider("Intercall Interval (ms)", 0, 2000, 500, step=50)

# 2. Per-call controls in a row
st.markdown("### Per-Call Parameters")
call_cols = st.columns(num_call)
call_params = []
for i, col in enumerate(call_cols):
    with col:
        st.markdown(f"**Call {i+1}**")
        waveform = st.selectbox(
            "Waveform", ['sine', 'square', 'triangle', 'sawtooth'],
            key=f"waveform_{i}"
        )
        f0 = st.slider(
            "Base Freq (Hz)", 100, 2000, 440, step=10, key=f"f0_{i}"
        )
        amplitude = st.slider(
            "Amplitude", 0.1, 1.0, 0.6, step=0.05, key=f"amp_{i}"
        )
        duration = st.slider(
            "Duration (ms)", 100, 5000, 1000, step=100, key=f"dur_{i}"
        )
        contour = st.selectbox(
            "Contour", ['flat', 'up', 'down'],
            key=f"contour_{i}"
        )
        sweep_range = st.slider(
            "Sweep", 1.0, 5.0, 3.0, step=0.1, key=f"sweep_{i}"
        )
        num_harmonics = st.slider(
            "Harmonics", 1, 8, 5, key=f"harm_{i}"
        )
        num_formant = st.slider(
            "Formants", 1, 4, 2, key=f"formant_{i}"
        )
        # --- Vibrato controls ---
        vibrato = st.checkbox("Add Vibrato", key=f"vib_{i}")
        if vibrato:
            vib_rate = st.slider("Vibrato Rate (Hz)", 1, 20, 5, key=f"vib_rate_{i}")
            vib_depth = st.slider("Vibrato Depth (Hz)", 1, 200, 20, key=f"vib_depth_{i}")
        else:
            vib_rate = 0
            vib_depth = 0
        call_params.append({
            'waveform': waveform,
            'f0': f0,
            'amplitude': amplitude,
            'duration': duration,
            'contour': contour,
            'sweep_range': sweep_range,
            'num_harmonics': num_harmonics,
            'num_formant': num_formant,
            'vibrato': vibrato,
            'vib_rate': vib_rate,
            'vib_depth': vib_depth
        })


# 3. Generate audio and spectrogram using the collected parameters
sequence = []
for i, params in enumerate(call_params):
    # Frequency sweep
    if params['contour'] == 'up':
        f_start = params['f0']
        f_end = params['f0'] * params['sweep_range']
    elif params['contour'] == 'down':
        f_start = params['f0']
        f_end = params['f0'] / params['sweep_range']
        f_end = max(20, f_end)
    else:
        f_start = f_end = params['f0']

    audio_data = generate_waveform_with_harmonics(
        params['waveform'], f_start, f_end, params['amplitude'], params['duration'],
        sample_rate, params['num_harmonics'],
        vibrato=params.get('vibrato', False),
        vib_rate=params.get('vib_rate', 0),
        vib_depth=params.get('vib_depth', 0)
    )

    np.random.seed(i)
    formant_freqs = [np.random.uniform(400, 3000) for _ in range(params['num_formant'])]
    formant_bandwidths = [np.random.uniform(100, 400) for _ in range(params['num_formant'])]
    for formant_freq, bandwidth in zip(formant_freqs, formant_bandwidths):
        audio_data = bandpass_filter(audio_data, formant_freq, bandwidth, sample_rate)

    sequence.append(audio_data)

    if i < num_call - 1:
        silence_samples = int(sample_rate * intercall_interval_ms / 1000)
        silence = np.zeros(silence_samples, dtype=np.int16)
        sequence.append(silence)


full_sequence = np.concatenate(sequence)
# Normalize the full sequence
max_abs = np.max(np.abs(full_sequence))
if max_abs > 0:
    full_sequence = full_sequence / max_abs
# Convert to int16 for WAV/audio
audio_int16 = np.int16(full_sequence * 32767)
# Calculate total duration in seconds
total_duration = len(audio_int16) / sample_rate

# 4. Show spectrogram and audio player
# Create ticks every 0.25 seconds, including start and end
tick_interval = 0.25
xticks = np.arange(0, total_duration, tick_interval)
# Ensure the last tick is exactly at the end
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

# Set x-ticks and labels
ax.set_xticks(xticks)
ax.set_xticklabels([f"{tick:.2f}" for tick in xticks])
st.pyplot(fig)

buf = io.BytesIO()
wavfile.write(buf, sample_rate, audio_int16)
st.audio(buf.getvalue(), format='audio/wav')

st.download_button(
    label="Save as WAV",
    data=buf.getvalue(),
    file_name="synthesized_sound.wav",
    mime="audio/wav"
)
