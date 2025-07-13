# Video Analysis Tool (Flask)

This project provides a web-based tool for analyzing YouTube videos using face analysis, scene classification, and color palette extraction. The backend is built with Flask and the UI is a simple HTML form.

## Features
- Download YouTube videos
- Extract frames at custom FPS and time range
- Analyze faces (age, gender, emotion) with DeepFace
- Scene classification with CLIP
- Color palette extraction
- Downloadable results (video, summary CSV, frame analysis CSV)

## Parameters Explained
- **Start Time (s):** The time (in seconds) from which to begin extracting frames from the video.
- **End Time (s):** The time (in seconds) at which to stop extracting frames from the video.
- **FPS (Frames Per Second):** The number of frames to extract per second from the selected video segment. Higher FPS means more frames and more detailed analysis, but also longer processing time.

## Setup

### 1. Clone the repository
```
git clone https://github.com/ankitbansal2101/Video-Analysis
cd Video-Analysis
```

### 2. Install dependencies
It is recommended to use a virtual environment (venv or conda).

```
pip install -r requirements.txt
```

### 3. Install system dependencies
- **ffmpeg** (for frame extraction)
- **yt-dlp** (for YouTube download, already in requirements.txt)

On macOS (with Homebrew):
```
brew install ffmpeg
```
On Ubuntu:
```
sudo apt-get install ffmpeg
```

### 4. Run the Flask server
```
python app.py
```

The server will start at [http://localhost:5000](http://localhost:5000).

## Usage
1. Open your browser and go to [http://localhost:5000](http://localhost:5000)
2. Enter the YouTube URL, start time, end time, and FPS.
3. Click "Start Analysis" and watch the progress bar.
4. Download the results when ready.

## Notes
- The first run may take longer as models are downloaded.
- For large videos or high FPS, processing may take several minutes.
- All results are saved in the server's working directory.

## Troubleshooting
- If you see errors about numpy or binary incompatibility, ensure you are using numpy 1.x (see requirements.txt).
- If you get CORS errors, make sure you are using the built-in HTML form (not React).
- For any issues, check the terminal output for error messages.

## License
MIT 
