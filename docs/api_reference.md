# API Reference

## Base URL
```
http://localhost:8000/api
```

## Endpoints

### Upload

#### POST `/api/upload/video`
Upload a video file for processing.

**Request**: `multipart/form-data`
- `file`: Video file (.mp4, .avi, .mkv, .webm, .mov)

**Response**:
```json
{
  "file_id": "uuid",
  "filename": "original_name.mp4",
  "path": "./data/uploads/uuid.mp4",
  "size_mb": 25.4
}
```

#### POST `/api/upload/audio`
Upload a reference audio file for voice cloning.

**Request**: `multipart/form-data`
- `file`: Audio file (.wav, .mp3, .flac, .ogg, .m4a)

**Response**: Same as video upload.

---

### Process

#### POST `/api/process`
Start a dubbing job.

**Request Body**:
```json
{
  "video_path": "./data/uploads/uuid.mp4",
  "audio_path": "./data/uploads/uuid.wav",
  "vlm_mode": "api",
  "tts_engine": "f5-tts",
  "source_lang": "English",
  "target_lang": "Vietnamese",
  "keep_original_audio": false
}
```

**Response**:
```json
{
  "job_id": "uuid",
  "status": "queued",
  "message": "Dubbing job queued."
}
```

---

### Status

#### GET `/api/status/{job_id}`
Check job progress.

**Response**:
```json
{
  "job_id": "uuid",
  "status": "processing",
  "progress": {
    "step": "vlm_translation",
    "current": 4,
    "total": 8,
    "percent": 50
  }
}
```

---

### Download

#### GET `/api/result/{job_id}`
Download the dubbed video.

#### GET `/api/result/{job_id}/srt`
Download the generated SRT subtitle file.
