import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter, resample_poly
import json
import os
import argparse
import sys
from pathlib import Path
from math import gcd
import gc

def bandpass_filter(data, center_freq, bandwidth, fs, order=2):
    if fs <= 0 or len(data) == 0:
        return data
    nyq = 0.5 * fs
    if bandwidth <= 0:
        return data
    low  = max(1e-5, (center_freq - bandwidth / 2) / nyq)
    high = min(0.99999, (center_freq + bandwidth / 2) / nyq)
    if low >= high:
        low = max(1e-5, high - 0.01)
    if low <= 0 or high <= 0 or low >= high:
        return data
    try:
        b, a = butter(order, [low, high], btype='band')
        return lfilter(b, a, data)
    except Exception:
        return data

def highpass_filter(data, cutoff_freq, fs, order=2):
    """High-pass filter for noise components"""
    if fs <= 0 or len(data) == 0:
        return data
    nyq = 0.5 * fs
    high = min(0.99999, cutoff_freq / nyq)
    if high <= 0:
        return data
    try:
        b, a = butter(order, high, btype='high')
        return lfilter(b, a, data)
    except Exception:
        return data

def apply_fadeout(audio: np.ndarray, fade_ms: float, sample_rate: int) -> np.ndarray:
    """Cosine fade-out — eliminates clicks, pops, and noise tails at the end."""
    fade_samples = min(int(fade_ms * sample_rate / 1000), len(audio))
    if fade_samples <= 0:
        return audio
    fade_curve = 0.5 * (1 + np.cos(np.pi * np.linspace(0, 1, fade_samples)))
    result = audio.copy()
    result[-fade_samples:] *= fade_curve
    return result

def apply_fadein(audio: np.ndarray, fade_ms: float, sample_rate: int) -> np.ndarray:
    """Cosine fade-in — eliminates clicks at the beginning."""
    fade_samples = min(int(fade_ms * sample_rate / 1000), len(audio))
    if fade_samples <= 0:
        return audio
    fade_curve = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, fade_samples)))
    result = audio.copy()
    result[:fade_samples] *= fade_curve
    return result

def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample audio from orig_sr to target_sr.
    Uses resample_poly (polyphase, anti-aliased) when ratio is reasonable,
    falls back to linear interpolation for extreme ratios.
    """
    if orig_sr == target_sr or len(audio) == 0:
        return audio
    try:
        g    = gcd(target_sr, orig_sr)
        up   = target_sr // g
        down = orig_sr   // g
        if max(up, down) <= 500:
            return resample_poly(audio, up, down).astype(np.float32)
    except Exception:
        pass
    # Fallback: linear interpolation
    old_length = len(audio)
    new_length = int(old_length * target_sr / orig_sr)
    if new_length <= 0:
        return audio
    return np.interp(
        np.linspace(0, old_length - 1, new_length),
        np.arange(old_length),
        audio
    ).astype(np.float32)

def get_inharmonic_frequency(partial_idx, fundamental, inharmonicity=0.1):
    """Generate inharmonic frequencies for more natural sounds"""
    harmonic_freq = fundamental * (partial_idx + 1)
    inharmonic_offset = harmonic_freq * inharmonicity * np.random.uniform(-0.5, 0.5)
    return max(50.0, harmonic_freq + inharmonic_offset)

def frequency_sweep(base_freq, t, sweep_rate=0, sweep_depth=0):
    """Add frequency sweeps and glides"""
    if sweep_rate > 0:
        sweep = sweep_depth * t * sweep_rate
        return base_freq + sweep
    return base_freq

def complex_amplitude_modulation(t, base_amp, trill_rate=0, trill_depth=0, flutter_rate=0, flutter_depth=0):
    """Generate complex amplitude modulation patterns"""
    modulation = 1.0
    if trill_rate > 0:
        modulation *= (1 + trill_depth * np.sin(2 * np.pi * trill_rate * t))
    if flutter_rate > 0:
        modulation *= (1 + flutter_depth * 0.1 * np.sin(2 * np.pi * flutter_rate * t))
    noise_mod = 1 + 0.02 * np.random.normal(0, 1, len(t))
    modulation *= noise_mod
    return base_amp * np.abs(modulation)

def add_breath_noise(audio, noise_level=0.05, filter_freq=8000, sample_rate=44100):
    """Add filtered noise to simulate breath sounds, faded out at the end."""
    if noise_level <= 0:
        return audio
    noise = np.random.normal(0, noise_level, len(audio))
    filtered_noise = highpass_filter(noise, filter_freq, sample_rate)
    # Fade noise out over last 30ms so it doesn't stick out at the end
    filtered_noise = apply_fadeout(filtered_noise, fade_ms=30, sample_rate=sample_rate)
    return audio + filtered_noise

def enhanced_envelope(t, attack_ms, decay_ms, total_ms, envelope_type='exponential'):
    """Generate various envelope shapes for different vocal characteristics"""
    n = len(t)
    if n == 0:
        return np.ones_like(t)

    total_ms  = max(float(total_ms),  1.0)
    attack_ms = max(float(attack_ms), 0.0)
    decay_ms  = max(float(decay_ms),  0.0)

    total_s  = total_ms  / 1000.0
    attack_s = attack_ms / 1000.0
    decay_s  = decay_ms  / 1000.0

    attack_samples  = min(int(attack_s / total_s * n), n)
    decay_samples   = min(int(decay_s  / total_s * n), n - attack_samples)
    sustain_samples = max(0, n - attack_samples - decay_samples)

    env = np.ones(n)

    if attack_samples > 0:
        if envelope_type == 'sharp_attack':
            attack_curve = 1 - np.exp(-10 * np.linspace(0, 1, attack_samples))
        elif envelope_type == 'smooth_swell':
            attack_curve = np.sin(np.pi * np.linspace(0, 0.5, attack_samples))
        else:
            attack_curve = 1 - np.exp(-5 * np.linspace(0, 1, attack_samples))
        env[:attack_samples] = attack_curve

    if sustain_samples > 0:
        sustain_start = attack_samples
        sustain_end   = attack_samples + sustain_samples
        if envelope_type == 'smooth_swell':
            sustain_variation = 0.05 * np.sin(
                2 * np.pi * 3 * np.linspace(0, 1, sustain_samples)
            )
            env[sustain_start:sustain_end] = 1.0 + sustain_variation
        else:
            env[sustain_start:sustain_end] = 1.0

    if decay_samples > 0:
        decay_start = attack_samples + sustain_samples
        if envelope_type == 'sharp_attack':
            decay_curve = np.exp(-5 * np.linspace(0, 1, decay_samples))
        else:
            decay_curve = np.exp(-3 * np.linspace(0, 1, decay_samples))
        env[decay_start:decay_start + decay_samples] = decay_curve

    return env

def additive_sines_with_enhanced_features(partials, duration_ms, sample_rate,
                                          inharmonicity=0.1, global_sweep_rate=0,
                                          global_trill_rate=0, noise_level=0.02):
    """Enhanced additive synthesis with natural vocal characteristics"""
    duration_ms = max(float(duration_ms), 10.0)
    n_samples = int(sample_rate * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, n_samples, endpoint=False)
    wave = np.zeros_like(t)

    for i, p in enumerate(partials):
        freq = float(p.get('freq', 50.0))
        if freq < 50.0:
            freq = 50.0

        if p.get('inharmonic', False):
            freq = get_inharmonic_frequency(i, freq, inharmonicity)

        sweep_rate         = float(global_sweep_rate) + float(p.get('sweep_rate', 0))
        sweep_depth        = freq * 0.2
        instantaneous_freq = frequency_sweep(freq, t, sweep_rate, sweep_depth)

        freq_jitter         = np.random.normal(0, freq * 0.01, len(t))
        freq_jitter         = np.cumsum(freq_jitter) * 0.0005
        instantaneous_freq += freq_jitter

        phase = 2 * np.pi * np.cumsum(instantaneous_freq) / sample_rate

        trill_rate  = float(global_trill_rate) + float(p.get('vib_rate', 0))
        trill_depth = float(p.get('vib_depth', 20)) / freq
        flutter_rate = trill_rate * 0.3

        modulated_amp = complex_amplitude_modulation(
            t, p['amp'], trill_rate, trill_depth, flutter_rate, 0.1
        )

        env = enhanced_envelope(
            t,
            float(p.get('attack', 5.0)),
            float(p.get('decay',  20.0)),
            duration_ms,
            p.get('envelope_type', 'exponential'),
        )

        sine_wave = np.sin(phase)

        distortion = float(p.get('distortion', 0))
        if distortion > 0:
            sine_wave = np.tanh(sine_wave * (1 + distortion))

        wave += modulated_amp * sine_wave * env

    wave = add_breath_noise(wave, noise_level, 8000, sample_rate)

    max_abs = np.max(np.abs(wave))
    if max_abs > 0:
        wave = wave / max_abs

    return wave

VOCAL_PRESETS = {
    "melodic": {
        "inharmonicity": 0.05,
        "global_sweep_rate": 0.5,
        "global_trill_rate": 12,
        "noise_level": 0.01,
        "formants": [(800, 200), (1200, 150), (2600, 200)]
    },
    "percussive": {
        "inharmonicity": 0.3,
        "global_sweep_rate": 0.1,
        "global_trill_rate": 25,
        "noise_level": 0.05,
        "formants": [(1000, 400), (2000, 300)]
    },
    "breathy": {
        "inharmonicity": 0.15,
        "global_sweep_rate": 0.2,
        "global_trill_rate": 8,
        "noise_level": 0.08,
        "formants": [(600, 300), (1800, 400), (3200, 500)]
    },
    "harmonic": {
        "inharmonicity": 0.02,
        "global_sweep_rate": 0.8,
        "global_trill_rate": 15,
        "noise_level": 0.005,
        "formants": [(900, 100), (1400, 120), (2800, 150)]
    }
}

def synthesize_single_sound(parameters):
    """Synthesize a single sound from parameters dictionary"""
    sample_rate  = max(int(parameters["global"].get("sample_rate", 44100)), 1000)
    duration     = max(float(parameters["global"].get("duration",   100)),  10.0)
    num_partials = max(int(parameters["global"].get("num_partials",   1)),   0)
    num_formants = max(int(parameters["global"].get("num_formants",   0)),   0)

    vocal_style  = parameters["global"].get("vocal_style", "melodic")
    preset       = VOCAL_PRESETS.get(vocal_style, VOCAL_PRESETS["melodic"])

    inharmonicity     = float(parameters["global"].get("inharmonicity",     preset["inharmonicity"]))
    global_sweep_rate = float(parameters["global"].get("global_sweep_rate", preset["global_sweep_rate"]))
    global_trill_rate = float(parameters["global"].get("global_trill_rate", preset["global_trill_rate"]))
    noise_level       = float(parameters["global"].get("noise_level",       preset["noise_level"]))

    # Cap noise level — above 0.08 it's audible hiss, not breath texture
    noise_level = min(noise_level, 0.08)

    partials = []
    for i in range(num_partials):
        if i < len(parameters["partials"]):
            p = parameters["partials"][i]

            attack = max(float(p.get('attack', 5.0)),  5.0)   # min 5ms
            decay  = max(float(p.get('decay',  20.0)), 20.0)  # min 20ms

            # Prevent attack+decay from consuming the entire duration
            if attack + decay > duration * 0.95:
                scale  = (duration * 0.95) / (attack + decay)
                attack *= scale
                decay  *= scale

            partials.append({
                'freq':          max(float(p.get('freq',   50.0)), 50.0),  # min 50Hz
                'amp':           max(float(p.get('amp',     0.5)),  0.0),
                'attack':        attack,
                'decay':         decay,
                'vibrato':       bool(p.get('vibrato', False)),
                'vib_rate':      float(p.get('vib_rate',  0.0)) if p.get('vibrato') else 0.0,
                'vib_depth':     float(p.get('vib_depth', 0.0)) if p.get('vibrato') else 0.0,
                'inharmonic':    bool(p.get('inharmonic', True)),
                'envelope_type': p.get('envelope_type', 'exponential'),
                'distortion':    max(float(p.get('distortion', 0.0)), 0.0),
                'sweep_rate':    float(p.get('sweep_rate', 0.0)),
            })

    if not partials:
        silence = np.zeros(int(sample_rate * duration / 1000))
        return silence, sample_rate

    audio_data = additive_sines_with_enhanced_features(
        partials, duration, sample_rate, inharmonicity,
        global_sweep_rate, global_trill_rate, noise_level
    )

    for i in range(num_formants):
        if i < len(parameters["formants"]):
            f           = parameters["formants"][i]
            center_freq = max(float(f.get('freq',      500.0)), 80.0)  # min 80Hz — prevents lowpass artifact
            bandwidth   = max(float(f.get('bandwidth',   0.0)),  0.0)  # 0 = skip
            audio_data  = bandpass_filter(audio_data, center_freq, bandwidth, sample_rate)

    if vocal_style in VOCAL_PRESETS:
        for formant_freq, bandwidth in preset["formants"]:
            audio_data = bandpass_filter(audio_data, formant_freq, bandwidth, sample_rate)

    # Per-sound fade-in and fade-out
    audio_data = apply_fadein( audio_data, fade_ms=5,  sample_rate=sample_rate)
    audio_data = apply_fadeout(audio_data, fade_ms=50, sample_rate=sample_rate)

    max_abs = np.max(np.abs(audio_data))
    if max_abs > 0:
        audio_data = audio_data / max_abs

    return audio_data, sample_rate

def process_concatenation_json(parameters, output_path):
    """Process a JSON file that contains concatenation instructions"""
    try:
        concat_params      = parameters.get("concatenation", {})
        sound_files        = concat_params.get("sound_files", [])
        intervals          = concat_params.get("intervals", [500])
        order              = concat_params.get("order", list(range(len(sound_files))))
        target_sample_rate = concat_params.get("sample_rate", 44100)

        if not sound_files:
            return False

        audio_segments = []
        for i, sound_config in enumerate(sound_files):
            if sound_config is None:
                continue

            audio_data, sample_rate = synthesize_single_sound(sound_config)

            if sample_rate != target_sample_rate:
                audio_data = resample_audio(audio_data, sample_rate, target_sample_rate)

            audio_segments.append((i, audio_data))

        if not audio_segments:
            return False

        sorted_segments = []
        for order_idx in order:
            if 0 <= order_idx < len(audio_segments):
                sorted_segments.append(audio_segments[order_idx])

        if not sorted_segments:
            return False

        concatenated_audio = []

        while len(intervals) < len(sorted_segments) - 1:
            intervals.append(intervals[-1] if intervals else 500)

        for i, (orig_idx, audio_segment) in enumerate(sorted_segments):
            concatenated_audio.extend(audio_segment)
            if i < len(sorted_segments) - 1:
                interval_samples = int(intervals[i] * target_sample_rate / 1000)
                concatenated_audio.extend(np.zeros(interval_samples))

        concatenated_audio = np.array(concatenated_audio)

        # Final fade-out on the entire WAV — removes any remaining tail noise
        concatenated_audio = apply_fadeout(
            concatenated_audio, fade_ms=80, sample_rate=target_sample_rate
        )

        max_abs = np.max(np.abs(concatenated_audio))
        if max_abs > 0:
            concatenated_audio = concatenated_audio / max_abs

        audio_int16 = np.int16(concatenated_audio * 32767)
        wavfile.write(output_path, target_sample_rate, audio_int16)

        return True

    except Exception as e:
        print(f"Error processing {output_path}: {str(e)}")
        return False

def find_json_files_recursive(input_dir):
    """Recursively find all JSON files in the input directory"""
    input_path = Path(input_dir)
    json_files = []
    try:
        for json_file in input_path.rglob("*.json"):
            if json_file.is_file():
                json_files.append(json_file)
    except Exception as e:
        print(f"Error scanning directory {input_dir}: {str(e)}")
        return []
    return sorted(json_files)

def get_relative_output_path(json_file_path, input_dir, output_dir):
    """Calculate the corresponding output path maintaining directory structure"""
    input_path = Path(input_dir).resolve()
    json_path  = Path(json_file_path).resolve()
    output_path = Path(output_dir).resolve()
    try:
        relative_path        = json_path.relative_to(input_path)
        output_relative_path = relative_path.with_suffix('.wav')
        return output_path / output_relative_path
    except ValueError:
        return output_path / f"{json_path.stem}.wav"

def process_single_json(json_path, output_path, verbose=False):
    """Process a single JSON file and create corresponding WAV file"""
    try:
        if verbose:
            print(f"Processing: {json_path}")

        with open(json_path, 'r') as f:
            parameters = json.load(f)

        if "concatenation" not in parameters:
            print(f"Error: {json_path} - Only concatenation format is supported")
            return False

        output_path.parent.mkdir(parents=True, exist_ok=True)

        success = process_concatenation_json(parameters, str(output_path))

        if success and verbose:
            print(f"Success: Generated {output_path}")
        elif not success:
            print(f"Failed: Could not process {json_path}")

        return success

    except Exception as e:
        print(f"Error processing {json_path}: {str(e)}")
        return False

def process_directory_recursive(input_dir, output_dir, verbose=False, max_depth=None):
    """Process all JSON files recursively in the input directory"""
    input_path  = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        print(f"Error: Input directory '{input_dir}' does not exist")
        return False

    if not input_path.is_dir():
        print(f"Error: Input path '{input_dir}' is not a directory")
        return False

    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error creating output directory '{output_dir}': {str(e)}")
        return False

    json_files = find_json_files_recursive(input_dir)

    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return True

    print(f"Found {len(json_files)} JSON files to process")

    success_count = 0
    total_count   = len(json_files)

    for i, json_file in enumerate(json_files, 1):
        output_file_path = get_relative_output_path(json_file, input_dir, output_dir)

        if verbose:
            print(f"[{i}/{total_count}] Processing: {json_file}")
            print(f"              Output to: {output_file_path}")

        if max_depth is not None:
            try:
                relative_path = json_file.relative_to(Path(input_dir))
                depth = len(relative_path.parts) - 1
                if depth > max_depth:
                    if verbose:
                        print(f"              Skipping: Exceeds max depth {max_depth}")
                    continue
            except ValueError:
                pass

        if process_single_json(json_file, output_file_path, verbose=False):
            success_count += 1
            if not verbose:
                print(f"[{i}/{total_count}] ✓ {json_file.name} -> {output_file_path.name}")
        else:
            print(f"[{i}/{total_count}] ✗ Failed: {json_file}")

        gc.collect()

    print(f"\nProcessing complete: {success_count}/{total_count} files processed successfully")
    return success_count == total_count

def create_example_json(output_path="example.json"):
    """Create an example JSON file showing format"""
    example = {
        "concatenation": {
            "sound_files": [
                {
                    "global": {
                        "sample_rate": 44100,
                        "duration": 500,
                        "num_partials": 2,
                        "num_formants": 1,
                        "vocal_style": "melodic",
                        "inharmonicity": 0.05,
                        "global_sweep_rate": 0.5,
                        "global_trill_rate": 12,
                        "noise_level": 0.01
                    },
                    "partials": [
                        {
                            "freq": 440,
                            "amp": 0.7,
                            "attack": 50,
                            "decay": 200,
                            "vibrato": True,
                            "vib_rate": 5,
                            "vib_depth": 20,
                            "inharmonic": True,
                            "envelope_type": "exponential",
                            "distortion": 0
                        },
                        {
                            "freq": 880,
                            "amp": 0.4,
                            "attack": 30,
                            "decay": 150,
                            "vibrato": False,
                            "vib_rate": 0,
                            "vib_depth": 0,
                            "inharmonic": True,
                            "envelope_type": "exponential",
                            "distortion": 0
                        }
                    ],
                    "formants": [
                        {
                            "freq": 800,
                            "bandwidth": 200
                        }
                    ]
                }
            ],
            "intervals": [0],
            "order": [0],
            "sample_rate": 44100
        }
    }

    with open(output_path, 'w') as f:
        json.dump(example, f, indent=2)

    print(f"Example JSON file created: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description='Audio Synthesis CLI Tool with Recursive Processing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input_folder output_folder
  %(prog)s input_folder output_folder --verbose
  %(prog)s input_folder output_folder --max-depth 2
  %(prog)s --create-example
        """
    )

    parser.add_argument('input_dir',  nargs='?', help='Directory containing input JSON files')
    parser.add_argument('output_dir', nargs='?', help='Directory for output WAV files')
    parser.add_argument('--create-example', action='store_true',
                        help='Create an example JSON file and exit')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--max-depth', type=int, metavar='N',
                        help='Maximum recursion depth (default: unlimited)')

    args = parser.parse_args()

    if args.create_example:
        create_example_json()
        return

    if not args.input_dir or not args.output_dir:
        parser.error('input_dir and output_dir are required unless using --create-example')

    print(f"Audio Synthesis Tool - Recursive Processing")
    print(f"Input directory:  {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    if args.max_depth is not None:
        print(f"Maximum depth: {args.max_depth}")
    print(f"Verbose mode: {'ON' if args.verbose else 'OFF'}")
    print("-" * 50)

    success = process_directory_recursive(
        args.input_dir,
        args.output_dir,
        verbose=args.verbose,
        max_depth=args.max_depth
    )

    if success:
        print("All files processed successfully!")
        sys.exit(0)
    else:
        print("Some files failed to process.")
        sys.exit(1)

if __name__ == "__main__":
    main()