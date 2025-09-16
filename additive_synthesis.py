import gradio as gr
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter, spectrogram
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import tempfile
import json
import os
import gc

# Set matplotlib to close figures automatically
plt.rcParams['figure.max_open_warning'] = 0

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

def get_default_frequency(partial_idx):
    """Calculate default frequency for a partial, ensuring it stays within bounds"""
    base_freq = 220 * (partial_idx + 1)
    return min(base_freq, 15000)  # Cap at 15000 Hz maximum

def synthesize_single_sound(parameters):
    """Synthesize a single sound from parameters dictionary"""
    # Extract parameters
    sample_rate = parameters["global"]["sample_rate"]
    duration = parameters["global"]["duration"]
    num_partials = parameters["global"]["num_partials"]
    num_formants = parameters["global"]["num_formants"]
    
    # Build partials list
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
                'vib_depth': p['vib_depth'] if p['vibrato'] else 0
            })
    
    # Synthesize audio
    audio_data = additive_sines_with_vibrato_and_envelope(partials, duration, sample_rate)
    
    # Apply formants
    for i in range(num_formants):
        if i < len(parameters["formants"]):
            f = parameters["formants"][i]
            audio_data = bandpass_filter(audio_data, f['freq'], f['bandwidth'], sample_rate)
    
    # Normalize
    max_abs = np.max(np.abs(audio_data))
    if max_abs > 0:
        audio_data = audio_data / max_abs
    
    return audio_data, sample_rate

def concatenate_sounds(json_files, intervals, order_indices, target_sample_rate=44100):
    """Concatenate multiple sounds with specified intervals and order"""
    try:
        # Filter out None files and load parameters
        valid_files = [(i, f) for i, f in enumerate(json_files) if f is not None]
        if not valid_files:
            return None, None, "No valid JSON files provided"
        
        # Load all parameter sets
        parameter_sets = []
        for orig_idx, file_path in valid_files:
            try:
                with open(file_path, 'r') as f:
                    params = json.load(f)
                parameter_sets.append((orig_idx, params))
            except Exception as e:
                return None, None, f"Error loading file {orig_idx + 1}: {str(e)}"
        
        # Generate audio for each parameter set
        audio_segments = []
        for orig_idx, params in parameter_sets:
            audio_data, sample_rate = synthesize_single_sound(params)
            
            # Resample if necessary (simple method)
            if sample_rate != target_sample_rate:
                # Simple resampling by interpolation
                old_length = len(audio_data)
                new_length = int(old_length * target_sample_rate / sample_rate)
                audio_data = np.interp(
                    np.linspace(0, old_length - 1, new_length),
                    np.arange(old_length),
                    audio_data
                )
            
            audio_segments.append((orig_idx, audio_data))
        
        # Sort by order indices
        sorted_segments = []
        for i, order_idx in enumerate(order_indices[:len(audio_segments)]):
            if 0 <= order_idx < len(audio_segments):
                sorted_segments.append(audio_segments[order_idx])
        
        if not sorted_segments:
            return None, None, "Invalid order configuration"
        
        # Concatenate with intervals
        concatenated_audio = []
        interval_samples_list = [int(interval_ms * target_sample_rate / 1000) for interval_ms in intervals]
        
        for i, (orig_idx, audio_segment) in enumerate(sorted_segments):
            concatenated_audio.extend(audio_segment)
            
            # Add interval (except after the last segment)
            if i < len(sorted_segments) - 1:
                interval_idx = min(i, len(interval_samples_list) - 1)
                interval_samples = interval_samples_list[interval_idx]
                concatenated_audio.extend(np.zeros(interval_samples))
        
        concatenated_audio = np.array(concatenated_audio)
        
        # Final normalization
        max_abs = np.max(np.abs(concatenated_audio))
        if max_abs > 0:
            concatenated_audio = concatenated_audio / max_abs
        
        return concatenated_audio, target_sample_rate, "Concatenation successful!"
        
    except Exception as e:
        return None, None, f"Concatenation failed: {str(e)}"

def create_concatenation_interface():
    """Create the concatenation page interface"""
    with gr.Column():
        gr.Markdown("# Sound Concatenation")
        gr.Markdown("Load up to 6 JSON parameter files and concatenate them with adjustable intervals.")
        
        # File upload section
        gr.Markdown("## Load Parameter Files")
        json_files = []
        for i in range(6):
            file_input = gr.File(
                label=f"JSON File {i + 1}",
                file_types=[".json"],
                value=None
            )
            json_files.append(file_input)
        
        # Configuration section
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Concatenation Settings")
                
                # Order configuration
                gr.Markdown("### Playback Order (0-5, -1 for skip)")
                order_inputs = []
                for i in range(6):
                    order_input = gr.Number(
                        label=f"Position {i + 1}",
                        value=i if i < 3 else -1,
                        precision=0,
                        minimum=-1,
                        maximum=5
                    )
                    order_inputs.append(order_input)
                
                # Interval configuration
                gr.Markdown("### Intervals Between Sounds (ms)")
                interval_inputs = []
                for i in range(5):  # 5 intervals for 6 sounds
                    interval_input = gr.Number(
                        label=f"Interval {i + 1} → {i + 2}",
                        value=500,
                        minimum=0,
                        maximum=5000
                    )
                    interval_inputs.append(interval_input)
        
        # Control buttons
        with gr.Row():
            concatenate_btn = gr.Button("Concatenate Sounds", variant="primary", size="lg")
            clear_btn = gr.Button("Clear All", variant="secondary")
        
        # Status and results
        concat_status = gr.Textbox(label="Status", interactive=False)
        
        with gr.Row():
            with gr.Column():
                concat_plot = gr.Plot(label="Concatenated Spectrogram")
                concat_audio = gr.Audio(label="Concatenated Audio", type="filepath")
        
        # Download section
        download_concat = gr.File(label="Download Concatenated Audio", visible=False)
        
        def perform_concatenation(*args):
            """Handle the concatenation process"""
            json_files_list = args[:6]
            intervals_list = args[6:11]
            order_list = [int(x) for x in args[11:17]]
            
            # Filter valid order indices
            valid_orders = [x for x in order_list if x >= 0]
            
            # Perform concatenation
            audio_data, sample_rate, status = concatenate_sounds(
                json_files_list, intervals_list, valid_orders
            )
            
            if audio_data is None:
                return gr.update(), gr.update(), status, gr.update(visible=False)
            
            # Create spectrogram
            plt.close('all')
            gc.collect()
            
            try:
                # Convert to int16 for spectrogram and saving
                audio_int16 = np.int16(audio_data * 32767)
                
                # Create spectrogram
                f_spec, t_spec, Sxx = spectrogram(
                    audio_int16.astype(float), fs=sample_rate, nperseg=1024
                )
                dominant_freqs = f_spec[np.argmax(Sxx, axis=0)]
                
                # Create plot
                fig = plt.figure(figsize=(16, 6))
                ax = fig.add_subplot(111)
                
                pcm = ax.pcolormesh(t_spec, f_spec, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
                fig.colorbar(pcm, ax=ax, label='dB')
                ax.plot(t_spec, dominant_freqs, color='w', linewidth=1.5, label='Dominant Freq')
                ax.set_ylabel('Frequency [Hz]')
                ax.set_xlabel('Time [sec]')
                ax.set_title('Concatenated Audio Spectrogram')
                ax.set_ylim(0, 15000)
                ax.legend()
                
                total_duration = len(audio_int16) / sample_rate
                tick_interval = max(0.25, total_duration / 20)  # Adaptive tick interval
                xticks = np.arange(0, total_duration, tick_interval)
                if not np.isclose(xticks[-1], total_duration):
                    xticks = np.append(xticks, total_duration)
                ax.set_xticks(xticks)
                ax.set_xticklabels([f"{tick:.2f}" for tick in xticks])
                
                plt.tight_layout()
                
                # Save audio file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                wavfile.write(temp_file.name, sample_rate, audio_int16)
                temp_file.close()
                
                # Create download file
                download_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav', prefix='concatenated_audio_')
                wavfile.write(download_file.name, sample_rate, audio_int16)
                download_file.close()
                
                return fig, temp_file.name, status, gr.update(value=download_file.name, visible=True)
                
            except Exception as e:
                return gr.update(), gr.update(), f"Error creating output: {str(e)}", gr.update(visible=False)
        
        def clear_all():
            """Clear all inputs"""
            updates = []
            # Clear file inputs
            for _ in range(6):
                updates.append(gr.update(value=None))
            # Reset intervals
            for _ in range(5):
                updates.append(gr.update(value=500))
            # Reset order
            for i in range(6):
                updates.append(gr.update(value=i if i < 3 else -1))
            # Clear outputs and status
            updates.extend([
                gr.update(),  # plot
                gr.update(),  # audio
                gr.update(value="Cleared all inputs"),  # status
                gr.update(visible=False)  # download
            ])
            return updates
        
        # Connect the concatenation function
        all_concat_inputs = json_files + interval_inputs + order_inputs
        concatenate_btn.click(
            fn=perform_concatenation,
            inputs=all_concat_inputs,
            outputs=[concat_plot, concat_audio, concat_status, download_concat]
        )
        
        # Connect the clear function
        all_concat_outputs = json_files + interval_inputs + order_inputs + [concat_plot, concat_audio, concat_status, download_concat]
        clear_btn.click(
            fn=clear_all,
            outputs=all_concat_outputs
        )

def export_parameters(sample_rate_idx, duration, num_partials, num_formants, *all_params):
    """Export all parameters to a JSON file"""
    try:
        # Extract sample rate value
        sample_rates = [32000, 44100, 48000]
        sample_rate_value = sample_rates[sample_rate_idx]
        
        # Organize parameters
        parameters = {
            "global": {
                "sample_rate": sample_rate_value,
                "sample_rate_idx": sample_rate_idx,
                "duration": duration,
                "num_partials": num_partials,
                "num_formants": num_formants
            },
            "partials": [],
            "formants": []
        }
        
        # Extract partial parameters (20 partials * 7 params each)
        for i in range(20):
            base_idx = i * 7
            partial_data = {
                "freq": all_params[base_idx],
                "amp": all_params[base_idx + 1],
                "attack": all_params[base_idx + 2],
                "decay": all_params[base_idx + 3],
                "vibrato": all_params[base_idx + 4],
                "vib_rate": all_params[base_idx + 5],
                "vib_depth": all_params[base_idx + 6]
            }
            parameters["partials"].append(partial_data)
        
        # Extract formant parameters (5 formants * 2 params each)
        formant_start_idx = 140
        for i in range(5):
            formant_data = {
                "freq": all_params[formant_start_idx + i * 2],
                "bandwidth": all_params[formant_start_idx + i * 2 + 1]
            }
            parameters["formants"].append(formant_data)
        
        # Create a temporary file for download
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json', prefix='synthesis_params_')
        json.dump(parameters, temp_file, indent=2)
        temp_file.close()
        
        return temp_file.name, "Parameters exported successfully!"
        
    except Exception as e:
        return None, f"Export failed: {str(e)}"

def import_parameters(file_path):
    """Import parameters from a JSON file and return update values for all components"""
    try:
        if file_path is None:
            return [gr.update() for _ in range(154)] + ["Please select a file to import."]
        
        with open(file_path, 'r') as f:
            parameters = json.load(f)
        
        # Prepare updates list for all components
        updates = []
        
        # Global parameters updates
        sample_rates = ["32000 Hz", "44100 Hz", "48000 Hz"]
        sample_rate_idx = parameters["global"]["sample_rate_idx"]
        # Convert index to actual dropdown choice string
        sample_rate_choice = sample_rates[sample_rate_idx]
        updates.append(gr.update(value=sample_rate_choice))  # sample_rate dropdown
        updates.append(gr.update(value=parameters["global"]["duration"]))  # duration
        updates.append(gr.update(value=parameters["global"]["num_partials"]))  # num_partials
        updates.append(gr.update(value=parameters["global"]["num_formants"]))  # num_formants
        
        # Partial parameters updates (20 partials * 7 params = 140 updates)
        for i in range(20):
            partial = parameters["partials"][i]
            updates.append(gr.update(value=partial["freq"]))
            updates.append(gr.update(value=partial["amp"]))
            updates.append(gr.update(value=partial["attack"]))
            updates.append(gr.update(value=partial["decay"]))
            updates.append(gr.update(value=partial["vibrato"]))
            updates.append(gr.update(value=partial["vib_rate"]))
            updates.append(gr.update(value=partial["vib_depth"]))
        
        # Formant parameters updates (5 formants * 2 params = 10 updates)
        for i in range(5):
            formant = parameters["formants"][i]
            updates.append(gr.update(value=formant["freq"]))
            updates.append(gr.update(value=formant["bandwidth"]))
        
        # Success message
        updates.append("Parameters imported successfully!")
        
        return updates
        
    except Exception as e:
        # Return no updates and error message
        return [gr.update() for _ in range(154)] + [f"Import failed: {str(e)}"]

def synthesize_audio(sample_rate_idx, duration, num_partials, num_formants, 
                    # Partial parameters (20 partials * 7 params each)
                    *partial_and_formant_params):
    
    # Clean up any existing figures before creating new ones
    plt.close('all')
    gc.collect()  # Force garbage collection
    
    # Map sample rate index to actual value
    sample_rates = [32000, 44100, 48000]
    sample_rate = sample_rates[sample_rate_idx]
    
    # Extract partial parameters (20 partials * 7 parameters each = 140 params)
    partials = []
    for i in range(20):
        if i < num_partials:
            base_idx = i * 7
            freq = partial_and_formant_params[base_idx]
            amp = partial_and_formant_params[base_idx + 1]
            attack = partial_and_formant_params[base_idx + 2]
            decay = partial_and_formant_params[base_idx + 3]
            vibrato = partial_and_formant_params[base_idx + 4]
            vib_rate = partial_and_formant_params[base_idx + 5] if vibrato else 0
            vib_depth = partial_and_formant_params[base_idx + 6] if vibrato else 0
            
            partials.append({
                'freq': freq,
                'amp': amp,
                'attack': attack,
                'decay': decay,
                'vibrato': vibrato,
                'vib_rate': vib_rate,
                'vib_depth': vib_depth
            })
    
    # Extract formant parameters (5 formants * 2 params each = 10 params)
    formant_freqs = []
    formant_bandwidths = []
    formant_start_idx = 140  # After 20 partials * 7 params
    
    for i in range(5):
        if i < num_formants:
            f_freq = partial_and_formant_params[formant_start_idx + i * 2]
            f_bw = partial_and_formant_params[formant_start_idx + i * 2 + 1]
            formant_freqs.append(f_freq)
            formant_bandwidths.append(f_bw)
    
    # Synthesize audio
    audio_data = additive_sines_with_vibrato_and_envelope(partials, duration, sample_rate)
    for f, bw in zip(formant_freqs, formant_bandwidths):
        audio_data = bandpass_filter(audio_data, f, bw, sample_rate)
    
    # Normalize
    max_abs = np.max(np.abs(audio_data))
    if max_abs > 0:
        audio_data = audio_data / max_abs
    audio_int16 = np.int16(audio_data * 32767)
    total_duration = len(audio_int16) / sample_rate
    
    # Create spectrogram
    f_spec, t_spec, Sxx = spectrogram(audio_int16.astype(float), fs=sample_rate, nperseg=1024)
    dominant_freqs = f_spec[np.argmax(Sxx, axis=0)]
    
    # Create plot with proper memory management
    fig = None
    try:
        # Create a new figure with explicit cleanup
        fig = plt.figure(figsize=(14, 4))
        ax = fig.add_subplot(111)
        
        pcm = ax.pcolormesh(t_spec, f_spec, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
        fig.colorbar(pcm, ax=ax, label='dB')
        ax.plot(t_spec, dominant_freqs, color='w', linewidth=1.5, label='Dominant Freq')
        ax.set_ylabel('Frequency [Hz]')
        ax.set_xlabel('Time [sec]')
        ax.set_title('Spectrogram (with Dominant Frequency)')
        ax.set_ylim(0, 15000)
        ax.legend()
        
        tick_interval = 0.25
        xticks = np.arange(0, total_duration, tick_interval)
        if not np.isclose(xticks[-1], total_duration):
            xticks = np.append(xticks, total_duration)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{tick:.2f}" for tick in xticks])
        
        # Make sure the plot is ready
        plt.tight_layout()
        
    except Exception as e:
        print(f"Plot creation error: {e}")
        # Create a simple fallback plot
        if fig:
            plt.close(fig)
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, f'Plot Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Spectrogram (Error in generation)')
    
    # Save audio to temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    wavfile.write(temp_file.name, sample_rate, audio_int16)
    temp_file.close()
    
    # Return the figure (Gradio will handle its lifecycle)
    return fig, temp_file.name

def create_synthesis_interface():
    """Create the basic synthesis interface"""
    with gr.Column():
        gr.Markdown("# Additive Vocal Synthesis with Formants")
        
        # Export/Import section
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## Parameter Management")
                with gr.Row():
                    export_btn = gr.Button("Export Parameters", variant="secondary")
                    import_btn = gr.Button("Import Parameters", variant="secondary")
                
                import_file = gr.File(
                    label="Select parameter file to import",
                    file_types=[".json"],
                    visible=False
                )
                
                download_file = gr.File(
                    label="Download Parameters",
                    visible=False
                )
                
                status_msg = gr.Textbox(
                    label="Status",
                    interactive=False,
                    visible=False
                )
        
        # Global controls
        with gr.Row():
            with gr.Column(scale=1):
                sample_rate = gr.Dropdown(
                    choices=["32000 Hz", "44100 Hz", "48000 Hz"],
                    value="44100 Hz",
                    label="Sample Rate",
                    type="index"
                )
                duration = gr.Slider(
                    minimum=100,
                    maximum=5000,
                    value=1000,
                    step=100,
                    label="Total Duration (ms)"
                )
                num_partials = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=3,
                    step=1,
                    label="Number of Partials (Sines)"
                )
                num_formants = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=2,
                    step=1,
                    label="Number of Formants"
                )
        
        # Partial controls - organized in rows of 5 columns each
        gr.Markdown("## Partial Parameters")
        
        partial_components = []
        partial_column_components = []
        
        # Create 4 rows of 5 columns each for 20 partials
        for row in range(4):
            with gr.Row() as partial_row:
                row_columns = []
                for col in range(5):
                    partial_idx = row * 5 + col
                    visible = partial_idx < 3  # Initially show first 3
                    
                    with gr.Column(visible=visible, scale=1) as column_comp:
                        gr.Markdown(f"**Partial {partial_idx + 1}**")
                        
                        freq = gr.Slider(
                            minimum=50,
                            maximum=15000,
                            value=get_default_frequency(partial_idx),
                            step=10,
                            label=f"Frequency (Hz)"
                        )
                        amp = gr.Slider(
                            minimum=0.05,
                            maximum=1.0,
                            value=0.5,
                            step=0.05,
                            label=f"Amplitude"
                        )
                        attack = gr.Slider(
                            minimum=0,
                            maximum=1000,
                            value=50,
                            step=10,
                            label=f"Attack (ms)"
                        )
                        decay = gr.Slider(
                            minimum=0,
                            maximum=1000,
                            value=200,
                            step=10,
                            label=f"Decay (ms)"
                        )
                        vibrato = gr.Checkbox(label="Vibrato", value=False)
                        vib_rate = gr.Slider(
                            minimum=1,
                            maximum=40,
                            value=5,
                            step=1,
                            label="Vib Rate",
                            visible=False
                        )
                        vib_depth = gr.Slider(
                            minimum=1,
                            maximum=300,
                            value=20,
                            step=1,
                            label="Vib Depth (Hz)",
                            visible=False
                        )
                        
                        # Store components for this partial
                        partial_components.extend([freq, amp, attack, decay, vibrato, vib_rate, vib_depth])
                        row_columns.append(column_comp)
                        
                        # Show/hide vibrato controls
                        vibrato.change(
                            fn=lambda x: (gr.update(visible=x), gr.update(visible=x)),
                            inputs=[vibrato],
                            outputs=[vib_rate, vib_depth]
                        )
                
                partial_column_components.extend(row_columns)
        
        # Formant controls
        gr.Markdown("## Formant Parameters")
        formant_components = []
        formant_row_components = []
        
        for i in range(5):
            visible = i < 2  # Initially show first 2
            with gr.Row(visible=visible) as formant_row:
                with gr.Column():
                    gr.Markdown(f"**Formant {i + 1}**")
                    f_freq = gr.Slider(
                        minimum=200,
                        maximum=15000,
                        value=500 + 500 * i,
                        step=10,
                        label=f"Frequency (Hz)"
                    )
                    f_bw = gr.Slider(
                        minimum=50,
                        maximum=1000,
                        value=200,
                        step=10,
                        label=f"Bandwidth (Hz)"
                    )
                    formant_components.extend([f_freq, f_bw])
                    formant_row_components.append(formant_row)
        
        # Output section
        gr.Markdown("## Output")
        with gr.Row():
            with gr.Column():
                plot_output = gr.Plot(label="Spectrogram")
                audio_output = gr.Audio(label="Generated Audio", type="filepath")
        
        # Dynamic UI update functions
        def update_partial_visibility(num_partials_val):
            updates = []
            for i in range(20):
                updates.append(gr.update(visible=(i < num_partials_val)))
            return updates
        
        def update_formant_visibility(num_formants_val):
            updates = []
            for i in range(5):
                updates.append(gr.update(visible=(i < num_formants_val)))
            return updates
        
        # Connect dynamic visibility updates
        num_partials.change(
            fn=update_partial_visibility,
            inputs=[num_partials],
            outputs=partial_column_components
        )
        
        num_formants.change(
            fn=update_formant_visibility,
            inputs=[num_formants],
            outputs=formant_row_components
        )
        
        # Collect all inputs for synthesis and export/import
        all_inputs = [sample_rate, duration, num_partials, num_formants]
        all_inputs.extend(partial_components)  # 20 partials * 7 params = 140
        all_inputs.extend(formant_components)  # 5 formants * 2 params = 10
        
        # All components for updates (154 total: 4 global + 140 partial + 10 formant)
        all_components = [sample_rate, duration, num_partials, num_formants] + partial_components + formant_components
        
        # Export functionality
        def handle_export(*args):
            file_path, message = export_parameters(*args)
            if file_path:
                return gr.update(value=file_path, visible=True), gr.update(value=message, visible=True)
            else:
                return gr.update(visible=False), gr.update(value=message, visible=True)
        
        export_btn.click(
            fn=handle_export,
            inputs=all_inputs,
            outputs=[download_file, status_msg]
        )
        
        # Import functionality  
        def show_import_file():
            return gr.update(visible=True), gr.update(visible=True)
        
        import_btn.click(
            fn=show_import_file,
            outputs=[import_file, status_msg]
        )
        
        import_file.change(
            fn=import_parameters,
            inputs=[import_file],
            outputs=all_components + [status_msg]
        )
        
        # Auto-synthesize on any parameter change
        def setup_auto_synthesis():
            for component in [sample_rate, duration, num_partials, num_formants] + partial_components + formant_components:
                component.change(
                    fn=synthesize_audio,
                    inputs=all_inputs,
                    outputs=[plot_output, audio_output]
                )
        
        # Set up auto-synthesis
        setup_auto_synthesis()
        
        return all_inputs, [plot_output, audio_output]

def create_interface():
    """Create the main interface with tabs"""
    with gr.Blocks(title="Additive Vocal Synthesis Suite", css=".gradio-container {max-width: none !important}") as demo:
        
        with gr.Tabs():
            with gr.Tab("Synthesis"):
                synthesis_inputs, synthesis_outputs = create_synthesis_interface()
                
                # Initial synthesis when the interface loads
                demo.load(
                    fn=synthesize_audio,
                    inputs=synthesis_inputs,
                    outputs=synthesis_outputs
                )
            
            with gr.Tab("Concatenation"):
                create_concatenation_interface()
    
    return demo

# Launch the app
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)