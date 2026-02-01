# Karaoke Accuracy Monitor

A Python program that analyzes singing accuracy by comparing microphone input with a reference song using neural network-based pitch detection.

## Features

- **Fast Neural Network Pitch Detection**: Uses ultrastar_pitch for accurate and efficient pitch analysis
- **Real-time Microphone Recording**: Records your singing performance
- **Similarity Analysis**: Compares your performance with the original song
- **Detailed Scoring**: Provides breakdown of pitch accuracy, exact matches, timbre, energy, and key detection
- **Overall Grading**: S to F grade system based on performance

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the program:

```bash
python main.py
```

The program will:

1. Ask for the path to an MP3 file (or use the default `./california_gurls.mp3`)
2. Load and analyze the song using neural network pitch detection
3. Prompt you to record your singing
4. Compare your performance and display results

### Programmatic Usage

```python
from main import KaraokeAnalyzer

# Initialize analyzer (16000 Hz for ultrastar_pitch)
analyzer = KaraokeAnalyzer(sample_rate=16000)

# Load reference song
analyzer.load_song("path/to/song.mp3")

# Record from microphone (5 seconds)
recorded_audio = analyzer.record_audio(duration=5)

# Compare and get scores
scores, features = analyzer.compare_audio(recorded_audio)

# Display results
analyzer.display_results(scores)
```

## How It Works

### Audio Analysis

The program uses **ultrastar_pitch**, a neural network-based pitch detection system originally designed for Ultrastar Deluxe karaoke games. This provides:

1. **Neural Network Pitch Detection**: Deep learning model trained on karaoke database
2. **Pitch Classification**: 12-class pitch detection (C, C#, D, D#, E, F, F#, G, G#, A, A#, B)
3. **Stochastic Postprocessing**: Statistical correction based on musical key detection
4. **Fast Processing**: Much faster than traditional pitch detection algorithms

Additional features:

- **RMS Energy**: Overall volume/energy
- **Spectral Centroid**: Brightness of the sound
- **Zero Crossing Rate**: Voice activity detection

### Similarity Metrics

- **Pitch Accuracy (50%)**: Compares pitch class distributions
- **Exact Pitch Match (20%)**: Direct frame-by-frame pitch comparison
- **Timbre Similarity (15%)**: Compares spectral characteristics
- **Energy Similarity (10%)**: Compares volume levels
- **Key Match (5%)**: Checks if singing is in the correct musical key

### Grading Scale

- S (90-100%): Perfect!
- A (80-89%): Excellent!
- B (70-79%): Great!
- C (60-69%): Good
- D (50-59%): Keep practicing
- F (0-49%): More practice needed

## Technical Details

- **Sample Rate**: 16000 Hz (required by ultrastar_pitch)
- **Pitch Classes**: 12 (chromatic scale from C to B)
- **Neural Network**: ONNX model trained on karaoke database
- **Processing**: Chunked processing (2-second windows) to avoid memory issues
- **Postprocessing**: Statistical key detection and pitch correction

## Advantages Over Traditional Methods

- **Much Faster**: Neural network inference is significantly faster than librosa's pyin
- **Better Accuracy**: Trained on karaoke-specific data
- **Key-Aware**: Automatically detects song key and corrects pitch predictions
- **Lightweight**: Smaller memory footprint than librosa

## Troubleshooting

### Microphone Not Working

Make sure your system has permission for microphone access and that sounddevice can detect your input device.

### Import Errors

Ensure all dependencies are installed:

```bash
pip install --upgrade -r requirements.txt
```

### Audio Quality Issues

- Use a quiet environment
- Speak/sing clearly into the microphone
- Adjust recording duration for longer/shorter phrases

## Dependencies

- **ultrastar-pitch**: Neural network-based pitch detection for karaoke
- **sounddevice**: Microphone recording
- **soundfile**: Fast audio file I/O
- **numpy**: Numerical operations
- **scipy**: Scientific computing and signal processing

## Credits

This project uses [ultrastar_pitch](https://github.com/paradigmn/ultrastar_pitch) for pitch detection, a neural network-based pitch detection system designed for karaoke applications.
