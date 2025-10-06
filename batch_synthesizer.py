import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter, spectrogram
import json
import os
import argparse
import sys
from pathlib import Path
import gc

def bandpass_filter(data, center_freq, bandwidth, fs, order=2):
    nyq = 0.5 * fs
    low = max(1e-5, (center_freq - bandwidth/2) / nyq)
    high = min(0.99999, (center_freq + bandwidth/2) / nyq)
    if low >= high:
        low = max(1e-5, high - 0.01)
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

def highpass_filter(data, cutoff_freq, fs, order=2):
    """High-pass filter for noise components"""
    nyq = 0.5 * fs
    high = min(0.99999, cutoff_freq / nyq)
    b, a = butter(order, high, btype='high')
    return lfilter(b, a, data)

def get_inharmonic_frequency(partial_idx, fundamental, inharmonicity=0.1):
    """Generate inharmonic frequencies for more natural sounds"""
    harmonic_freq = fundamental * (partial_idx + 1)
    inharmonic_offset = harmonic_freq * inharmonicity * np.random.uniform(-0.5, 0.5)
    return max(50, harmonic_freq + inharmonic_offset)

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
    """Add filtered noise to simulate breath sounds"""
    if noise_level <= 0:
        return audio
    
    noise = np.random.normal(0, noise_level, len(audio))
    filtered_noise = highpass_filter(noise, filter_freq, sample_rate)
    return audio + filtered_noise

def enhanced_envelope(t, attack_ms, decay_ms, total_ms, envelope_type='exponential'):
    """Generate various envelope shapes for different vocal characteristics"""
    attack_samples = int((attack_ms / 1000) * len(t) / (total_ms / 1000))
    decay_samples = int((decay_ms / 1000) * len(t) / (total_ms / 1000))
    sustain_samples = max(0, len(t) - attack_samples - decay_samples)
    
    env = np.ones_like(t)
    
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
        sustain_end = attack_samples + sustain_samples
        sustain_level = 1.0
        if envelope_type == 'smooth_swell':
            sustain_variation = 0.05 * np.sin(2 * np.pi * 3 * np.linspace(0, 1, sustain_samples))
            env[sustain_start:sustain_end] = sustain_level + sustain_variation
        else:
            env[sustain_start:sustain_end] = sustain_level
    
    if decay_samples > 0:
        decay_start = len(t) - decay_samples
        if envelope_type == 'sharp_attack':
            decay_curve = np.exp(-5 * np.linspace(0, 1, decay_samples))
        else:
            decay_curve = np.exp(-3 * np.linspace(0, 1, decay_samples))
        env[decay_start:] = decay_curve
    
    return env

def additive_sines_with_enhanced_features(partials, duration_ms, sample_rate, 
                                        inharmonicity=0.1, global_sweep_rate=0, 
                                        global_trill_rate=0, noise_level=0.02):
    """Enhanced additive synthesis with natural vocal characteristics"""
    t = np.linspace(0, duration_ms / 1000, int(sample_rate * duration_ms / 1000), endpoint=False)
    wave = np.zeros_like(t)
    
    for i, p in enumerate(partials):
        if 'inharmonic' in p and p['inharmonic']:
            freq = get_inharmonic_frequency(i, p['freq'], inharmonicity)
        else:
            freq = p['freq']
        
        sweep_rate = global_sweep_rate + p.get('sweep_rate', 0)
        sweep_depth = freq * 0.2
        instantaneous_freq = frequency_sweep(freq, t, sweep_rate, sweep_depth)
        
        freq_jitter = np.random.normal(0, freq * 0.01, len(t))
        freq_jitter = np.cumsum(freq_jitter) * 0.0005
        instantaneous_freq += freq_jitter
        
        phase = 2 * np.pi * np.cumsum(instantaneous_freq) / sample_rate
        
        trill_rate = global_trill_rate + p.get('vib_rate', 0)
        trill_depth = p.get('vib_depth', 20) / freq if freq > 0 else 0
        flutter_rate = trill_rate * 0.3
        
        modulated_amp = complex_amplitude_modulation(
            t, p['amp'], trill_rate, trill_depth, flutter_rate, 0.1
        )
        
        envelope_type = p.get('envelope_type', 'exponential')
        env = enhanced_envelope(t, p['attack'], p['decay'], duration_ms, envelope_type)
        
        sine_wave = np.sin(phase)
        
        if p.get('distortion', 0) > 0:
            distortion_amount = p['distortion']
            sine_wave = np.tanh(sine_wave * (1 + distortion_amount))
        
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
    sample_rate = parameters["global"]["sample_rate"]
    duration = parameters["global"]["duration"]
    num_partials = parameters["global"]["num_partials"]
    num_formants = parameters["global"]["num_formants"]
    
    vocal_style = parameters["global"].get("vocal_style", "melodic")
    preset = VOCAL_PRESETS.get(vocal_style, VOCAL_PRESETS["melodic"])
    
    inharmonicity = parameters["global"].get("inharmonicity", preset["inharmonicity"])
    global_sweep_rate = parameters["global"].get("global_sweep_rate", preset["global_sweep_rate"])
    global_trill_rate = parameters["global"].get("global_trill_rate", preset["global_trill_rate"])
    noise_level = parameters["global"].get("noise_level", preset["noise_level"])
    
    partials = []
    for i in range(num_partials):
        if i < len(parameters["partials"]):
            p = parameters["partials"][i]
            partials.append({
                'freq': p['freq'],
                'amp': p['amp'],
                'attack': p['attack'],
                'decay': p['decay'],
                'vibrato': p['vibrato'],
                'vib_rate': p['vib_rate'] if p['vibrato'] else 0,
                'vib_depth': p['vib_depth'] if p['vibrato'] else 0,
                'inharmonic': p.get('inharmonic', True),
                'envelope_type': p.get('envelope_type', 'exponential'),
                'distortion': p.get('distortion', 0),
                'sweep_rate': p.get('sweep_rate', 0)
            })
    
    audio_data = additive_sines_with_enhanced_features(
        partials, duration, sample_rate, inharmonicity, 
        global_sweep_rate, global_trill_rate, noise_level
    )
    
    for i in range(num_formants):
        if i < len(parameters["formants"]):
            f = parameters["formants"][i]
            audio_data = bandpass_filter(audio_data, f['freq'], f['bandwidth'], sample_rate)
    
    if vocal_style in VOCAL_PRESETS:
        for formant_freq, bandwidth in preset["formants"]:
            audio_data = bandpass_filter(audio_data, formant_freq, bandwidth, sample_rate)
    
    max_abs = np.max(np.abs(audio_data))
    if max_abs > 0:
        audio_data = audio_data / max_abs
    
    return audio_data, sample_rate

def process_concatenation_json(parameters, output_path):
    """Process a JSON file that contains concatenation instructions"""
    try:
        concat_params = parameters.get("concatenation", {})
        sound_files = concat_params.get("sound_files", [])
        intervals = concat_params.get("intervals", [500])
        order = concat_params.get("order", list(range(len(sound_files))))
        target_sample_rate = concat_params.get("sample_rate", 44100)
        
        if not sound_files:
            return False
        
        audio_segments = []
        for i, sound_config in enumerate(sound_files):
            if sound_config is None:
                continue
                
            audio_data, sample_rate = synthesize_single_sound(sound_config)
            
            if sample_rate != target_sample_rate:
                old_length = len(audio_data)
                new_length = int(old_length * target_sample_rate / sample_rate)
                audio_data = np.interp(
                    np.linspace(0, old_length - 1, new_length),
                    np.arange(old_length),
                    audio_data
                )
            
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
        
        max_abs = np.max(np.abs(concatenated_audio))
        if max_abs > 0:
            concatenated_audio = concatenated_audio / max_abs
        
        audio_int16 = np.int16(concatenated_audio * 32767)
        wavfile.write(output_path, target_sample_rate, audio_int16)
        
        return True
        
    except Exception as e:
        print(f"Error processing {output_path}: {str(e)}")
        return False

def process_single_json(json_path, output_dir):
    """Process a single JSON file and create corresponding WAV file(s)"""
    try:
        with open(json_path, 'r') as f:
            parameters = json.load(f)
        
        # Only accept concatenation syntax
        if "concatenation" not in parameters:
            print(f"Error: {json_path} - Only concatenation format is supported")
            return False
        
        json_name = Path(json_path).stem
        output_path = Path(output_dir) / f"{json_name}.wav"
        
        return process_concatenation_json(parameters, str(output_path))
            
    except Exception as e:
        print(f"Error processing {json_path}: {str(e)}")
        return False

def process_directory(input_dir, output_dir):
    """Process all JSON files in the input directory"""
    input_path = Path(input_dir)
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
    
    json_files = list(input_path.glob("*.json"))
    
    if not json_files:
        return True
    
    success_count = 0
    total_count = len(json_files)
    
    for json_file in json_files:
        if process_single_json(str(json_file), str(output_path)):
            success_count += 1
        
        gc.collect()
    
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
            "intervals": [0],  # 0ms interval for single sound
            "order": [0],
            "sample_rate": 44100
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(example, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Audio Synthesis CLI Tool')
    parser.add_argument('input_dir', help='Directory containing input JSON files')
    parser.add_argument('output_dir', help='Directory for output WAV files')
    parser.add_argument('--create-example', action='store_true', help='Create an example JSON file')
    
    args = parser.parse_args()
    
    if args.create_example:
        create_example_json()
        return
    
    success = process_directory(args.input_dir, args.output_dir)
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()