# AI Sound Generation 

This repository contains interactive apps for additive sound synthesis and sound testing, plus command-line tools for batch file synthesis and file management for testing the results on participants:

- **additive_synthesis.py**: A Gradio app that allows mixing multiple sine waves together (additive synthesis) and concatenating up to 6 different sounds.
- **sound_test.py**: A Streamlit app for creating sound with a greater variety of parameters, but can't mix multiple sine waves.
- **audio_synthesis_tool.py**: A command-line batch synthesizer for processing multiple JSON parameter files recursively in a file structure.
- **file_processor.py**: A utility for batch file processing with unique ID generation and pairing documentation. Useful for testing results on participants.

Both interactive apps can be run independently using Docker and support importing/exporting parameter presets for creating realistic bird sounds and other complex audio.

---

## Requirements

- [Docker](https://docs.docker.com/get-started/get-docker/) installed on your system (for interactive apps)
- Docker daemon is running (for interactive apps)
- Python 3.7+ with required packages (for command-line tools)

### Python Dependencies (for command-line tools)

```bash
pip install numpy scipy matplotlib gradio streamlit
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/TediVirag/AI-sound-generation.git
```

### 2. Interactive Apps (Docker)

#### **Additive Synthesis App**

Build the Docker image:

```bash
docker build -f dockerfile.synthesis -t additive-synth-app .
```

Run the app:

```bash
docker run -p 7860:7860 additive-synth-app
```

#### **Sound Test App**

Build the Docker image:

```bash
docker build -f dockerfile.test -t sound-test-app .
```

Run the app:

```bash
docker run -p 8501:8501 sound-test-app
```
### **Access the Interactive Apps**

- **Additive Synthesis App**: [http://localhost:7860](http://localhost:7860)
- **Sound Test App**: [http://localhost:8501](http://localhost:8501)

---

### 3. Command-Line Tools

#### **Batch Audio Synthesis Tool**

Process multiple JSON parameter files to generate WAV audio files:

```bash
# Process all JSON files in input_folder and generate WAV files in output_folder
python batch_synthesizer.py input_folder output_folder

# With verbose output
python batch_synthesizer.py input_folder output_folder --verbose

# Limit recursion depth
python batch_synthesizer.py input_folder output_folder --max-depth 2

# Create an example JSON file to understand the format
python batch_synthesizer.py --create-example
```

**Features:**
- Recursive processing of nested directories
- Maintains directory structure in output
- Only supports concatenation JSON format
- Enhanced synthesis
- Memory-efficient batch processing

#### **File Processor Tool**

Batch process files with unique ID generation:

```bash
# Interactive mode
python test_file_processor.py

# Programmatic use
from test_file_processor import process_files_with_params
process_files_with_params("input_folder", "output_path", id_length=8)
```

**Features:**
- Recursively processes all files in a directory
- Adds unique IDs to original filenames
- Creates copies with ID-only names in output directory  
- Generates pairing document for tracking file relationships
- Configurable ID length (4-16 characters)
- Preserves original file metadata

---

## Sound Parameters Reference

The synthesis engine uses a comprehensive set of parameters to create realistic sounds:

### Global Parameters

**Basic Settings:**
- **sample_rate**: Audio sample rate in Hz (32000, 44100, or 48000) - determines audio quality and frequency range
- **duration**: Length of the generated sound in milliseconds
- **num_partials**: Number of frequency components (harmonics/overtones) to generate (1-20)
- **num_formants**: Number of formant filters to apply (1-5) - shapes the spectral envelope

**Enhanced Global Parameters:**
- **vocal_style**: Overall character ("melodic", "percussive", "harmonic") - affects how partials interact
- **inharmonicity**: Amount of deviation from perfect harmonic ratios (0.0-1.0) - adds natural imperfection
- **global_sweep_rate**: Speed of frequency modulation applied to all partials
- **global_trill_rate**: Speed of rapid amplitude/frequency variations (Hz)
- **noise_level**: Amount of background noise mixed in (0.0-1.0) - adds realism

### Partial Parameters (Per Harmonic Component)

Each of the 20 possible partials has these parameters:

**Frequency & Amplitude:**
- **freq**: Base frequency of this partial in Hz
- **amp**: Amplitude/volume of this partial (0.0-1.0)

**Envelope Shaping:**
- **attack**: Time in milliseconds for the sound to reach full amplitude
- **decay**: Time in milliseconds for the sound to fade out
- **envelope_type**: Shape of the amplitude envelope ("exponential", "sharp_attack", "smooth_swell")

**Modulation:**
- **vibrato**: Whether to apply vibrato (frequency wobble) - boolean
- **vib_rate**: Speed of vibrato in Hz
- **vib_depth**: Depth of vibrato as frequency deviation in Hz

**Harmonic Character:**
- **inharmonic**: Whether this partial deviates from harmonic ratios - boolean
- **distortion**: Amount of harmonic distortion to add (0.0-1.0)

### Formant Parameters (Spectral Filtering)

Each of the 5 possible formants has:
- **freq**: Center frequency of the formant filter in Hz - creates vocal-like resonances
- **bandwidth**: Width of the formant filter in Hz - controls how sharp or broad the resonance is

---

## Parameter Presets

The apps support importing and exporting parameter configurations as JSON files. This allows you to:

- Save complex sound configurations
- Share parameter sets for specific sounds
- Create multi-part sounds by concatenating different parameter files

## File Structure

```
├── additive_synthesis.py          # Main Gradio synthesis app
├── sound_test.py                  # Streamlit parameter testing app
├── batch_synthesizer.py           # Command-line batch synthesizer
├── test_file_processor.py         # File processing utility with unique IDs
├── requirements.txt               # Python dependencies
├── dockerfile.synthesis           # Docker config for synthesis app
├── dockerfile.test                # Docker config for test app
└── README.md
```

---

## Usage Tips

1. **Start Simple**: Begin with a few partials and basic parameters
2. **Use Presets**: Import example sound files to understand parameter relationships
3. **Experiment with Formants**: These create vocal-like character in sounds
4. **Try Inharmonicity**: Small amounts (0.1-0.2) add natural imperfection
5. **Combine Sounds**: Use the concatenation feature to create complex sequences
6. **Batch Processing**: Use the command-line tool for processing large numbers of parameter files
7. **File Processing for Testing**: Use the file processor to process large collections of audio files with unique tracking for testing purposes

---

## License

[MIT License](LICENSE)

---

## Acknowledgments

- Built with [Gradio](https://gradio.app/) and [Streamlit](https://streamlit.io/) for interactive web interfaces
- Uses [NumPy](https://numpy.org/), [SciPy](https://scipy.org/), and [Matplotlib](https://matplotlib.org/) for audio processing