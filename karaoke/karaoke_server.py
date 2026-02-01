#!/usr/bin/env python3
"""
Real-time Karaoke Scoring Backend Server
FastAPI + WebSocket server for Godot integration
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import asyncio
import numpy as np
import librosa
import soundfile as sf
import json
import io
import base64
import os
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass, asdict
import uuid


@dataclass
class PitchSegment:
    time: float
    frequency: float
    confidence: float


@dataclass
class ScoreUpdate:
    """Real-time score update sent to Godot"""
    timestamp: float
    current_score: float
    average_score: float
    speed_multiplier: float
    note_name: str
    rating: str
    is_singing: bool


class KaraokeSession:
    """Manages a single karaoke session"""
    
    def __init__(self, session_id: str, original_audio_path: str, 
                 pitch_tolerance: float = 50, sample_rate: int = 22050):
        self.session_id = session_id
        self.original_audio_path = original_audio_path
        self.pitch_tolerance = pitch_tolerance
        self.sample_rate = sample_rate
        
        # Load and analyze original song
        self.original_audio, _ = librosa.load(original_audio_path, sr=sample_rate)
        self.song_duration = len(self.original_audio) / sample_rate
        self.original_pitches = self._extract_pitch(self.original_audio)
        
        # Session state
        self.score_history = []
        self.start_time = None
        self.active = False
        
    def _extract_pitch(self, audio: np.ndarray) -> List[PitchSegment]:
        """Extract pitch from audio"""
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C6'),
            sr=self.sample_rate,
            hop_length=512
        )
        
        segments = []
        times = librosa.times_like(f0, sr=self.sample_rate, hop_length=512)
        
        for i, (t, freq, prob) in enumerate(zip(times, f0, voiced_probs)):
            if not np.isnan(freq) and voiced_flag[i]:
                segments.append(PitchSegment(
                    time=float(t),
                    frequency=float(freq),
                    confidence=float(prob)
                ))
        
        return segments
    
    @staticmethod
    def hz_to_cents(freq1: float, freq2: float) -> float:
        """Calculate difference in cents"""
        if freq1 <= 0 or freq2 <= 0:
            return float('inf')
        return 1200 * np.log2(freq2 / freq1)
    
    @staticmethod
    def hz_to_note_name(freq: float) -> str:
        """Convert frequency to note name"""
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        midi = 69 + 12 * np.log2(freq / 440.0)
        note_num = int(round(midi))
        octave = (note_num // 12) - 1
        note = notes[note_num % 12]
        return f"{note}{octave}"
    
    def score_audio_chunk(self, audio_chunk: bytes, chunk_timestamp: float) -> ScoreUpdate:
        """
        Score an audio chunk from Godot
        
        Args:
            audio_chunk: Raw audio bytes (PCM float32)
            chunk_timestamp: Time in song (seconds)
        
        Returns:
            ScoreUpdate object
        """
        # Convert bytes to numpy array
        audio_array = np.frombuffer(audio_chunk, dtype=np.float32)
        
        # Extract pitch from chunk
        user_pitches = self._extract_pitch_from_chunk(audio_array)
        
        # Score the chunk
        score = self._score_chunk(user_pitches, chunk_timestamp)
        
        # Update history
        self.score_history.append(score)
        if len(self.score_history) > 10:
            self.score_history.pop(0)
        
        # Calculate average
        avg_score = np.mean(self.score_history) if self.score_history else 0
        
        # Speed multiplier
        speed_multiplier = 0.5 + (avg_score / 100) * 1.5
        
        # Get note name
        note_name = ""
        is_singing = False
        if user_pitches:
            avg_freq = np.mean([p.frequency for p in user_pitches])
            note_name = self.hz_to_note_name(avg_freq)
            is_singing = True
        
        # Rating
        if score >= 90:
            rating = "PERFECT"
        elif score >= 75:
            rating = "GREAT"
        elif score >= 60:
            rating = "GOOD"
        elif score >= 40:
            rating = "OK"
        else:
            rating = "MISS"
        
        return ScoreUpdate(
            timestamp=chunk_timestamp,
            current_score=score,
            average_score=avg_score,
            speed_multiplier=speed_multiplier,
            note_name=note_name,
            rating=rating,
            is_singing=is_singing
        )
    
    def _extract_pitch_from_chunk(self, audio_chunk: np.ndarray) -> List[PitchSegment]:
        """Extract pitch from small audio chunk"""
        if len(audio_chunk) < 2048:
            return []
        
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio_chunk,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C6'),
                sr=self.sample_rate,
                hop_length=512
            )
            
            segments = []
            for i, (freq, prob) in enumerate(zip(f0, voiced_probs)):
                if not np.isnan(freq) and voiced_flag[i]:
                    segments.append(PitchSegment(
                        time=0,
                        frequency=float(freq),
                        confidence=float(prob)
                    ))
            
            return segments
        except:
            return []
    
    def _score_chunk(self, user_pitches: List[PitchSegment], chunk_time: float) -> float:
        """Score a chunk against original"""
        if not user_pitches:
            return 0.0
        
        # Get original pitches in time window
        window = 0.5  # 0.5 second window
        original_in_window = [
            p for p in self.original_pitches
            if chunk_time <= p.time <= chunk_time + window
        ]
        
        if not original_in_window:
            return 0.0
        
        scores = []
        for user_pitch in user_pitches:
            # Find closest original
            closest = min(
                original_in_window,
                key=lambda p: abs(p.time - chunk_time),
                default=None
            )
            
            if closest:
                cents_off = abs(self.hz_to_cents(closest.frequency, user_pitch.frequency))
                pitch_score = np.exp(-(cents_off ** 2) / (2 * (self.pitch_tolerance ** 2)))
                pitch_score = max(0.0, min(1.0, pitch_score))
                scores.append(pitch_score)
        
        return np.mean(scores) * 100 if scores else 0.0


# FastAPI app
app = FastAPI(title="Karaoke Scoring Server")

# Enable CORS for Godot
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active sessions
sessions = {}

# Songs directory
SONGS_DIR = Path("songs")
SONGS_DIR.mkdir(exist_ok=True)


@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "online",
        "service": "Karaoke Scoring Server",
        "active_sessions": len(sessions)
    }


@app.post("/upload_song")
async def upload_song(file: UploadFile = File(...)):
    """
    Upload a song file to the server
    Returns song_id for use in sessions
    """
    # Generate unique song ID
    song_id = str(uuid.uuid4())
    file_path = SONGS_DIR / f"{song_id}.wav"
    
    # Save uploaded file
    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)
    
    # Get song info
    audio, sr = librosa.load(str(file_path), sr=22050)
    duration = len(audio) / sr
    
    return {
        "song_id": song_id,
        "filename": file.filename,
        "duration": duration,
        "status": "uploaded"
    }


@app.get("/songs")
async def list_songs():
    """List all available songs"""
    songs = []
    for song_file in SONGS_DIR.glob("*.wav"):
        song_id = song_file.stem
        try:
            audio, sr = librosa.load(str(song_file), sr=22050)
            duration = len(audio) / sr
            songs.append({
                "song_id": song_id,
                "duration": duration
            })
        except:
            pass
    
    return {"songs": songs}


@app.post("/create_session")
async def create_session(song_id: str, pitch_tolerance: float = 50):
    """
    Create a new karaoke session
    
    Args:
        song_id: ID of the uploaded song
        pitch_tolerance: Tolerance in cents (default: 50)
    
    Returns:
        session_id for WebSocket connection
    """
    song_path = SONGS_DIR / f"{song_id}.wav"
    
    if not song_path.exists():
        return JSONResponse(
            status_code=404,
            content={"error": "Song not found"}
        )
    
    # Create session
    session_id = str(uuid.uuid4())
    session = KaraokeSession(
        session_id=session_id,
        original_audio_path=str(song_path),
        pitch_tolerance=pitch_tolerance
    )
    
    sessions[session_id] = session
    
    return {
        "session_id": session_id,
        "song_duration": session.song_duration,
        "pitch_tolerance": pitch_tolerance
    }


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session"""
    if session_id in sessions:
        del sessions[session_id]
        return {"status": "deleted"}
    
    return JSONResponse(
        status_code=404,
        content={"error": "Session not found"}
    )


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time scoring
    
    Godot sends: {"audio": base64_encoded_audio, "timestamp": 1.5}
    Server responds: ScoreUpdate JSON
    """
    await websocket.accept()
    
    # Get session
    session = sessions.get(session_id)
    if not session:
        await websocket.send_json({"error": "Session not found"})
        await websocket.close()
        return
    
    session.active = True
    
    try:
        while True:
            # Receive audio chunk from Godot
            data = await websocket.receive_json()
            
            # Handle different message types
            if "command" in data:
                if data["command"] == "start":
                    await websocket.send_json({"status": "started", "duration": session.song_duration})
                    continue
                elif data["command"] == "stop":
                    session.active = False
                    await websocket.send_json({"status": "stopped"})
                    break
            
            # Process audio chunk
            if "audio" in data and "timestamp" in data:
                # Decode base64 audio
                audio_bytes = base64.b64decode(data["audio"])
                timestamp = data["timestamp"]
                
                # Score the chunk
                score_update = session.score_audio_chunk(audio_bytes, timestamp)
                
                # Send score back to Godot
                await websocket.send_json(asdict(score_update))
    
    except WebSocketDisconnect:
        session.active = False
        print(f"Client disconnected from session {session_id}")
    except Exception as e:
        print(f"Error in WebSocket: {e}")
        await websocket.send_json({"error": str(e)})
    finally:
        session.active = False


@app.get("/session/{session_id}/stats")
async def get_session_stats(session_id: str):
    """Get statistics for a session"""
    session = sessions.get(session_id)
    if not session:
        return JSONResponse(
            status_code=404,
            content={"error": "Session not found"}
        )
    
    if not session.score_history:
        return {
            "average_score": 0,
            "max_score": 0,
            "min_score": 0,
            "total_chunks": 0
        }
    
    return {
        "average_score": np.mean(session.score_history),
        "max_score": max(session.score_history),
        "min_score": min(session.score_history),
        "total_chunks": len(session.score_history),
        "speed_multiplier": 0.5 + (np.mean(session.score_history) / 100) * 1.5
    }


if __name__ == "__main__":
    import uvicorn
    
    print("🎤 Karaoke Scoring Server")
    print("=" * 60)
    print("Starting server on http://localhost:8000")
    print("WebSocket endpoint: ws://localhost:8000/ws/{session_id}")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
