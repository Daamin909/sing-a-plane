#!/usr/bin/env python3
"""
Demo script that generates synthetic test audio to demonstrate the karaoke scorer
"""

import numpy as np
import soundfile as sf
from karaoke_scorer import KaraokeScorer, print_score_report


def generate_melody(notes, duration=0.5, sample_rate=22050):
    """
    Generate a simple melody from MIDI note numbers
    
    Args:
        notes: List of MIDI note numbers
        duration: Duration of each note in seconds
        sample_rate: Audio sample rate
    """
    audio = []
    
    for note in notes:
        # Convert MIDI to frequency
        freq = 440 * (2 ** ((note - 69) / 12))
        
        # Generate samples for this note
        samples = int(duration * sample_rate)
        t = np.linspace(0, duration, samples, False)
        
        # Create a simple sine wave with envelope
        envelope = np.exp(-3 * t)  # Decay envelope
        note_audio = np.sin(2 * np.pi * freq * t) * envelope * 0.3
        
        audio.append(note_audio)
    
    return np.concatenate(audio)


def add_noise(audio, noise_level=0.02):
    """Add a bit of noise to make it more realistic"""
    noise = np.random.normal(0, noise_level, len(audio))
    return audio + noise


def generate_test_files():
    """Generate test audio files for demonstration"""
    
    # Simple melody: C-D-E-F-G-A-B-C (1 octave)
    original_notes = [60, 62, 64, 65, 67, 69, 71, 72]
    
    print("Generating original song...")
    original_audio = generate_melody(original_notes, duration=0.8)
    original_audio = add_noise(original_audio, 0.01)
    sf.write('demo_original.wav', original_audio, 22050)
    print("✓ Saved: demo_original.wav")
    
    # Perfect performance (same notes)
    print("\nGenerating perfect performance...")
    perfect_audio = generate_melody(original_notes, duration=0.8)
    perfect_audio = add_noise(perfect_audio, 0.02)
    sf.write('demo_perfect.wav', perfect_audio, 22050)
    print("✓ Saved: demo_perfect.wav")
    
    # Good performance (slightly off-pitch, good timing)
    print("\nGenerating good performance...")
    good_notes = [60.3, 62.1, 63.8, 65.2, 67.1, 68.9, 71.2, 72.1]  # Slightly off
    good_audio = generate_melody(good_notes, duration=0.8)
    good_audio = add_noise(good_audio, 0.03)
    sf.write('demo_good.wav', good_audio, 22050)
    print("✓ Saved: demo_good.wav")
    
    # Mediocre performance (more off-pitch)
    print("\nGenerating mediocre performance...")
    mediocre_notes = [60.5, 61.5, 64.5, 66, 67.5, 70, 70.5, 71.5]  # Quite off
    mediocre_audio = generate_melody(mediocre_notes, duration=0.85)  # Slightly slower
    mediocre_audio = add_noise(mediocre_audio, 0.04)
    sf.write('demo_mediocre.wav', mediocre_audio, 22050)
    print("✓ Saved: demo_mediocre.wav")
    
    # Bad performance (very off-pitch and timing)
    print("\nGenerating bad performance...")
    bad_notes = [61, 63, 66, 64, 68, 71, 69, 73]  # Random-ish
    bad_audio = generate_melody(bad_notes, duration=0.6)  # Wrong timing
    bad_audio = add_noise(bad_audio, 0.05)
    sf.write('demo_bad.wav', bad_audio, 22050)
    print("✓ Saved: demo_bad.wav")


def run_demo():
    """Run scoring demo on all test files"""
    
    test_files = [
        ('demo_perfect.wav', 'Perfect Performance'),
        ('demo_good.wav', 'Good Performance'),
        ('demo_mediocre.wav', 'Mediocre Performance'),
        ('demo_bad.wav', 'Bad Performance')
    ]
    
    scorer = KaraokeScorer(
        pitch_tolerance_cents=50,
        timing_window_ms=100
    )
    
    results = []
    
    for test_file, description in test_files:
        print("\n" + "="*70)
        print(f"Testing: {description}")
        print("="*70)
        
        result = scorer.score_performance('demo_original.wav', test_file)
        print_score_report(result)
        
        # Create visualization
        viz_name = test_file.replace('.wav', '_analysis.png')
        scorer.visualize_comparison('demo_original.wav', test_file, result, viz_name)
        print(f"📊 Visualization saved: {viz_name}")
        
        results.append({
            'file': test_file,
            'description': description,
            'score': result.overall_score,
            'multiplier': result.speed_multiplier
        })
    
    # Summary comparison
    print("\n" + "="*70)
    print("SUMMARY COMPARISON")
    print("="*70)
    print(f"{'Performance':<20} {'Score':<10} {'Speed Multiplier':<15}")
    print("-"*70)
    for r in results:
        print(f"{r['description']:<20} {r['score']:>6.1f}/100  {r['multiplier']:>10.2f}x")
    print("="*70)


if __name__ == "__main__":
    import os
    import sys
    
    print("🎤 Karaoke Scorer Demo")
    print("="*70)
    
    # Generate test files
    print("\nStep 1: Generating test audio files...")
    generate_test_files()
    
    # Run demo
    print("\n\nStep 2: Running scorer on test files...")
    input("\nPress Enter to continue with scoring demo...")
    run_demo()
    
    print("\n\n✅ Demo complete!")
    print("\nGenerated files:")
    print("  • demo_original.wav - Original melody")
    print("  • demo_perfect.wav - Perfect performance")
    print("  • demo_good.wav - Good performance")
    print("  • demo_mediocre.wav - Mediocre performance")
    print("  • demo_bad.wav - Bad performance")
    print("  • *_analysis.png - Visualizations for each")
    print("\nTry it with your own audio:")
    print("  python karaoke_scorer.py your_song.wav your_recording.wav")