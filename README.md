# AI sound generation 

This repository contains two interactive Streamlit apps for additive sound synthesis and testing:

- **additive_synthesis.py**: An app that allows mixing multiple sine waves together(additive synthesis).
- **sound_test.py**: An app for creating sound with a greater variety of parameters(can't mix multiple sine waves though).

Both apps can be run independently using Docker.

---

## Requirements

- [Docker](https://docs.docker.com/get-started/get-docker/) installed on your system
- Docker deamon is running

---

## Getting Started

### 1. Clone the repository

### 2. Build and Run the Apps

#### **Additive Synthesis App**

Build the Docker image:

```bash
docker build -f dockerfile.synthesis -t additive-synth-app .
```

Run the app:

```bash
docker run -p 8501:8501 additive-synth-app
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

---

### 3. Access the App

Open your browser and go to:  
[http://localhost:8501](http://localhost:8501)

---

## File Structure

```

├── additive_synthesis.py
├── sound_test.py
├── requirements.txt
├── Dockerfile.additive
├── Dockerfile.soundtest
└── README.md
```

---

## License

[MIT License](LICENSE) (or specify your license here)

---

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Uses [NumPy](https://numpy.org/), [SciPy](https://scipy.org/), and [Matplotlib](https://matplotlib.org/)
