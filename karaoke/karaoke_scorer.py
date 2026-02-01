import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy import signal
from dataclasses import dataclass
from typing import List, Tuple, Optional
import json


@dataclass
class PitchSegment:
    """Represents a pitch measurement at a point in time"""
    time: float
    frequency: float  # Hz
    confidence: float  # 0-1


@dataclass
class ScoringResult:
    """Results from comparing user singing to original"""
    overall_score: float  # 0-100
    pitch_accuracy: float  # 0-100
    timing_score: float  # 0-100
    note_hit_rate: float  # percentage of notes hit correctly
    detailed_scores: List[dict]  # frame-by-frame breakdown
    speed_multiplier: float  # game speed boost based on performance


class KaraokeScorer:
    def __init__(self, 
                 pitch_tolerance_cents: float = 50,  # Half semitone tolerance
                 timing_window_ms: float = 100,
                 hop_length: int = 512,
                 sample_rate: int = 22050):
        """
        Initialize the karaoke scorer
        
        Args:
            pitch_tolerance_cents: How many cents off-pitch is acceptable (100 cents = 1 semitone)
            timing_window_ms: Time window for matching notes (milliseconds)
            hop_length: Samples between pitch measurements
            sample_rate: Audio sample rate
        """
        self.pitch_tolerance_cents = pitch_tolerance_cents
        self.timing_window_ms = timing_window_ms
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        
    def extract_pitch(self, audio_path: str) -> List[PitchSegment]:
        """
        Extract pitch from an audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of PitchSegment objects with time, frequency, and confidence
        """
        # Load audio
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Extract pitch using pYIN algorithm (good for singing)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),  # ~65 Hz
            fmax=librosa.note_to_hz('C6'),  # ~1047 Hz
            sr=sr,
            hop_length=self.hop_length
        )
        
        # Convert to time-stamped segments
        segments = []
        times = librosa.times_like(f0, sr=sr, hop_length=self.hop_length)
        
        for i, (time, freq, prob) in enumerate(zip(times, f0, voiced_probs)):
            if not np.isnan(freq) and voiced_flag[i]:
                segments.append(PitchSegment(
                    time=float(time),
                    frequency=float(freq),
                    confidence=float(prob)
                ))
        
        return segments
    
    @staticmethod
    def hz_to_cents(freq1: float, freq2: float) -> float:
        """
        Calculate the difference between two frequencies in cents
        100 cents = 1 semitone
        """
        if freq1 <= 0 or freq2 <= 0:
            return float('inf')
        return 1200 * np.log2(freq2 / freq1)
    
    @staticmethod
    def hz_to_midi(freq: float) -> float:
        """Convert frequency to MIDI note number"""
        return 69 + 12 * np.log2(freq / 440.0)
    
    def match_pitches(self, 
                      original_pitches: List[PitchSegment],
                      user_pitches: List[PitchSegment]) -> List[dict]:
        """
        Match user pitches to original pitches and score them
        
        Returns:
            List of matched segments with scores
        """
        matched_segments = []
        timing_window_s = self.timing_window_ms / 1000.0
        
        for orig_pitch in original_pitches:
            # Find user pitches within timing window
            candidates = [
                user_pitch for user_pitch in user_pitches
                if abs(user_pitch.time - orig_pitch.time) <= timing_window_s
            ]
            
            if not candidates:
                # No singing detected at this moment
                matched_segments.append({
                    'time': orig_pitch.time,
                    'original_freq': orig_pitch.frequency,
                    'user_freq': None,
                    'cents_off': None,
                    'pitch_score': 0.0,
                    'timing_score': 0.0,
                    'note_hit': False
                })
                continue
            
            # Find closest match in time
            best_match = min(candidates, key=lambda p: abs(p.time - orig_pitch.time))
            
            # Calculate pitch difference in cents
            cents_off = self.hz_to_cents(orig_pitch.frequency, best_match.frequency)
            
            # Score pitch accuracy (gaussian falloff)
            pitch_score = np.exp(-(cents_off ** 2) / (2 * (self.pitch_tolerance_cents ** 2)))
            pitch_score = max(0.0, min(1.0, pitch_score))
            
            # Score timing accuracy
            time_diff_ms = abs(best_match.time - orig_pitch.time) * 1000
            timing_score = 1.0 - (time_diff_ms / self.timing_window_ms)
            timing_score = max(0.0, min(1.0, timing_score))
            
            # Note is "hit" if within tolerance
            note_hit = abs(cents_off) <= self.pitch_tolerance_cents
            
            matched_segments.append({
                'time': orig_pitch.time,
                'original_freq': orig_pitch.frequency,
                'user_freq': best_match.frequency,
                'cents_off': cents_off,
                'pitch_score': pitch_score,
                'timing_score': timing_score,
                'note_hit': note_hit,
                'confidence': best_match.confidence
            })
        
        return matched_segments
    
    def calculate_overall_score(self, matched_segments: List[dict]) -> ScoringResult:
        """
        Calculate final scores from matched segments
        """
        if not matched_segments:
            return ScoringResult(0, 0, 0, 0, [], 1.0)
        
        # Calculate component scores
        pitch_scores = [s['pitch_score'] for s in matched_segments]
        timing_scores = [s['timing_score'] for s in matched_segments]
        note_hits = [s['note_hit'] for s in matched_segments]
        
        pitch_accuracy = np.mean(pitch_scores) * 100
        timing_score = np.mean(timing_scores) * 100
        note_hit_rate = (sum(note_hits) / len(note_hits)) * 100
        
        # Overall score (weighted combination)
        overall_score = (
            pitch_accuracy * 0.6 +  # Pitch is most important
            timing_score * 0.2 +    # Timing matters
            note_hit_rate * 0.2     # Hitting notes matters
        )
        
        # Calculate speed multiplier for game (1.0 to 2.0x)
        # Perfect score = 2x speed, 50% = 1x speed, below 50% = slower
        speed_multiplier = 0.5 + (overall_score / 100) * 1.5
        speed_multiplier = max(0.5, min(2.0, speed_multiplier))
        
        return ScoringResult(
            overall_score=overall_score,
            pitch_accuracy=pitch_accuracy,
            timing_score=timing_score,
            note_hit_rate=note_hit_rate,
            detailed_scores=matched_segments,
            speed_multiplier=speed_multiplier
        )
    
    def score_performance(self, 
                         original_audio_path: str,
                         user_audio_path: str) -> ScoringResult:
        """
        Main function to score a karaoke performance
        
        Args:
            original_audio_path: Path to original song
            user_audio_path: Path to user's recording
            
        Returns:
            ScoringResult with all scores and analysis
        """
        print("Extracting pitch from original song...")
        original_pitches = self.extract_pitch(original_audio_path)
        
        print("Extracting pitch from user recording...")
        user_pitches = self.extract_pitch(user_audio_path)
        
        print("Matching and scoring pitches...")
        matched_segments = self.match_pitches(original_pitches, user_pitches)
        
        print("Calculating final scores...")
        result = self.calculate_overall_score(matched_segments)
        
        return result
    
    def visualize_comparison(self,
                            original_audio_path: str,
                            user_audio_path: str,
                            result: ScoringResult,
                            output_path: str = 'karaoke_analysis.png'):
        """
        Create a visualization comparing original and user pitch
        """
        # Extract pitches again for visualization
        original_pitches = self.extract_pitch(original_audio_path)
        user_pitches = self.extract_pitch(user_audio_path)
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # Plot 1: Pitch comparison
        ax1 = axes[0]
        if original_pitches:
            orig_times = [p.time for p in original_pitches]
            orig_freqs = [self.hz_to_midi(p.frequency) for p in original_pitches]
            ax1.plot(orig_times, orig_freqs, 'b-', label='Original', linewidth=2, alpha=0.7)
        
        if user_pitches:
            user_times = [p.time for p in user_pitches]
            user_freqs = [self.hz_to_midi(p.frequency) for p in user_pitches]
            ax1.plot(user_times, user_freqs, 'r-', label='Your singing', linewidth=2, alpha=0.7)
        
        ax1.set_ylabel('MIDI Note Number')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_title('Pitch Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Score over time
        ax2 = axes[1]
        if result.detailed_scores:
            times = [s['time'] for s in result.detailed_scores]
            pitch_scores = [s['pitch_score'] * 100 for s in result.detailed_scores]
            ax2.plot(times, pitch_scores, 'g-', linewidth=2)
            ax2.fill_between(times, 0, pitch_scores, alpha=0.3, color='green')
            ax2.axhline(y=50, color='orange', linestyle='--', label='50% threshold')
            ax2.set_ylabel('Pitch Score (%)')
            ax2.set_xlabel('Time (seconds)')
            ax2.set_title('Accuracy Over Time')
            ax2.set_ylim(0, 100)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Pitch error (cents)
        ax3 = axes[2]
        if result.detailed_scores:
            times = [s['time'] for s in result.detailed_scores if s['cents_off'] is not None]
            cents = [s['cents_off'] for s in result.detailed_scores if s['cents_off'] is not None]
            
            colors = ['green' if abs(c) <= self.pitch_tolerance_cents else 'red' for c in cents]
            ax3.scatter(times, cents, c=colors, alpha=0.6, s=20)
            ax3.axhline(y=0, color='blue', linestyle='-', linewidth=2, label='Perfect pitch')
            ax3.axhline(y=self.pitch_tolerance_cents, color='orange', linestyle='--', 
                       label=f'±{self.pitch_tolerance_cents} cents tolerance')
            ax3.axhline(y=-self.pitch_tolerance_cents, color='orange', linestyle='--')
            ax3.set_ylabel('Pitch Error (cents)')
            ax3.set_xlabel('Time (seconds)')
            ax3.set_title('Pitch Accuracy (0 = perfect)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
        plt.close()


def print_score_report(result: ScoringResult):
    """Print a nice formatted score report"""
    print("\n" + "="*60)
    print("🎤 KARAOKE PERFORMANCE REPORT 🎤")
    print("="*60)
    print(f"\n🌟 Overall Score: {result.overall_score:.1f}/100")
    print(f"\n📊 Component Scores:")
    print(f"   • Pitch Accuracy:  {result.pitch_accuracy:.1f}/100")
    print(f"   • Timing Score:    {result.timing_score:.1f}/100")
    print(f"   • Notes Hit:       {result.note_hit_rate:.1f}%")
    print(f"\n🚀 Speed Multiplier: {result.speed_multiplier:.2f}x")
    
    # Performance rating
    if result.overall_score >= 90:
        rating = "🏆 SUPERSTAR! Amazing performance!"
    elif result.overall_score >= 80:
        rating = "⭐ EXCELLENT! You've got talent!"
    elif result.overall_score >= 70:
        rating = "👍 GREAT! Keep it up!"
    elif result.overall_score >= 60:
        rating = "😊 GOOD! You're getting there!"
    else:
        rating = "💪 KEEP PRACTICING! You'll improve!"
    
    print(f"\n{rating}")
    print("="*60 + "\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python karaoke_scorer.py <original_song.wav> <user_recording.wav>")
        print("\nExample:")
        print("  python karaoke_scorer.py original.wav my_singing.wav")
        sys.exit(1)
    
    original_file = sys.argv[1]
    user_file = sys.argv[2]
    
    # Initialize scorer
    scorer = KaraokeScorer(
        pitch_tolerance_cents=50,  # Half semitone tolerance
        timing_window_ms=100
    )
    
    # Score the performance
    result = scorer.score_performance(original_file, user_file)
    
    # Print results
    print_score_report(result)
    
    # Create visualization
    scorer.visualize_comparison(original_file, user_file, result)
    
    # Save detailed results to JSON
    detailed_results = {
        'overall_score': result.overall_score,
        'pitch_accuracy': result.pitch_accuracy,
        'timing_score': result.timing_score,
        'note_hit_rate': result.note_hit_rate,
        'speed_multiplier': result.speed_multiplier,
        'frame_count': len(result.detailed_scores),
        'frames': result.detailed_scores[:100]  # First 100 frames as sample
    }
    
    with open('karaoke_results.json', 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print("📁 Detailed results saved to karaoke_results.json")