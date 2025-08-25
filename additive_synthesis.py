import gradio as gr
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter, spectrogram
import matplotlib.pyplot as plt
import tempfile
import json
import os

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
    return min(base_freq, 2000)  # Cap at 2000 Hz maximum

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
    
    # Create plot
    plt.close('all')  # Close any existing plots
    fig, ax = plt.subplots(figsize=(14, 4))
    pcm = ax.pcolormesh(t_spec, f_spec, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
    fig.colorbar(pcm, ax=ax, label='dB')
    ax.plot(t_spec, dominant_freqs, color='w', linewidth=1.5, label='Dominant Freq')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [sec]')
    ax.set_title('Spectrogram (with Dominant Frequency)')
    ax.set_ylim(0, 5000)
    ax.legend()
    
    tick_interval = 0.25
    xticks = np.arange(0, total_duration, tick_interval)
    if not np.isclose(xticks[-1], total_duration):
        xticks = np.append(xticks, total_duration)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{tick:.2f}" for tick in xticks])
    
    # Save audio to temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    wavfile.write(temp_file.name, sample_rate, audio_int16)
    temp_file.close()
    
    return fig, temp_file.name

def create_interface():
    with gr.Blocks(title="Additive Vocal Synthesis with Formants", css=".gradio-container {max-width: none !important}") as demo:
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
                            maximum=2000,
                            value=get_default_frequency(partial_idx),  # Fixed default frequency
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
                            maximum=20,
                            value=5,
                            step=1,
                            label="Vib Rate",
                            visible=False
                        )
                        vib_depth = gr.Slider(
                            minimum=1,
                            maximum=200,
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
                        maximum=4000,
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
        
        # Initial synthesis when the interface loads
        demo.load(
            fn=synthesize_audio,
            inputs=all_inputs,
            outputs=[plot_output, audio_output]
        )
        
        # Set up auto-synthesis
        setup_auto_synthesis()
    
    return demo

# Launch the app
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)