import os
import glob
import threading
import uuid
import time
from flask import Flask, request, jsonify, send_file, render_template
import numpy as np
import pandas as pd
from PIL import Image
from deepface import DeepFace
from transformers import CLIPProcessor, CLIPModel
from sklearn.cluster import KMeans
import torch

app = Flask(__name__)

# In-memory job store
jobs = {}

# Helper: Analysis function
def run_analysis(job_id, video_url, start_time, end_time, fps):
    try:
        jobs[job_id]['status'] = 'downloading'
        jobs[job_id]['progress'] = 0.05
        video_path = f"job_{job_id}_video.mp4"
        os.system(f"yt-dlp -f best -o {video_path} {video_url}")

        jobs[job_id]['status'] = 'extracting_frames'
        jobs[job_id]['progress'] = 0.15
        os.makedirs(f"frames_{job_id}", exist_ok=True)
        totaltime = end_time - start_time
        os.system(f"ffmpeg -ss {start_time} -t {totaltime} -i {video_path} -vf fps={fps} frames_{job_id}/frame_%03d.jpg -hide_banner -loglevel error")
        frame_paths = sorted(glob.glob(f"frames_{job_id}/frame_*.jpg"))

        jobs[job_id]['status'] = 'deepface_analysis'
        jobs[job_id]['progress'] = 0.25
        frame_results = []
        for i, frame in enumerate(frame_paths):
            try:
                analysis = DeepFace.analyze(frame, actions=["age", "gender", "emotion"], enforce_detection=False)
                frame_results.append({
                    "frame": frame,
                    "age": analysis[0]["age"],
                    "gender": analysis[0]["gender"],
                    "emotion": analysis[0]["dominant_emotion"]
                })
            except Exception:
                frame_results.append({"frame": frame, "age": None, "gender": None, "emotion": None})
            jobs[job_id]['progress'] = 0.25 + 0.25 * (i+1)/len(frame_paths) if frame_paths else 0.5

        jobs[job_id]['status'] = 'clip_classification'
        jobs[job_id]['progress'] = 0.5
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        scene_labels = ["bedroom", "kitchen", "street", "mirror selfie", "dance", "food", "product demo"]
        def classify_scene(image_path):
            image = Image.open(image_path).convert("RGB")
            inputs = clip_processor(text=scene_labels, images=image, return_tensors="pt", padding=True)
            outputs = clip_model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1).detach().numpy().flatten()
            return scene_labels[np.argmax(probs)]
        for r in frame_results:
            r["scene"] = classify_scene(r["frame"])
        jobs[job_id]['progress'] = 0.65

        jobs[job_id]['status'] = 'palette_extraction'
        def extract_palette(img_path, k=5):
            img = Image.open(img_path).convert("RGB")
            data = np.array(img).reshape(-1, 3)
            kmeans = KMeans(n_clusters=k, n_init="auto").fit(data)
            return kmeans.cluster_centers_.astype(int)
        for r in frame_results:
            try:
                r["palette"] = extract_palette(r["frame"]).tolist()
            except:
                r["palette"] = None
        jobs[job_id]['progress'] = 0.8

        jobs[job_id]['status'] = 'summary'
        video_caption = "grwm for my first day back at uni 💄📚 #grwm #firstday #backtoschool"
        upload_time = "2024-09-01T08:00:00Z"
        hashtags = [w for w in video_caption.split() if w.startswith("#")]
        df = pd.DataFrame(frame_results)
        summary = {
            "avg_age": df["age"].dropna().mean(),
            "gender_mode": df["gender"].mode()[0] if df["gender"].notna().sum() > 0 else None,
            "dominant_emotion": df["emotion"].mode()[0] if df["emotion"].notna().sum() > 0 else None,
            "dominant_scene": df["scene"].mode()[0] if df["scene"].notna().sum() > 0 else None,
            
            "sample_palette": df["palette"].iloc[0] if df["palette"].notna().sum() > 0 else None,
        }
        df.to_csv(f"frame_analysis_{job_id}.csv", index=False)
        pd.DataFrame([summary]).to_csv(f"summary_{job_id}.csv", index=False)
        jobs[job_id]['progress'] = 1.0
        jobs[job_id]['status'] = 'done'
        jobs[job_id]['files'] = {
            'video': video_path,
            'summary': f"summary_{job_id}.csv",
            'frame_analysis': f"frame_analysis_{job_id}.csv"
        }
    except Exception as e:
        jobs[job_id]['status'] = 'error'
        jobs[job_id]['error'] = str(e)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json(silent=True)
    if not data:
        data = request.form
    video_url = data.get('url')
    start_time = int(data.get('start_time', 10))
    end_time = int(data.get('end_time', 100))
    fps = int(data.get('fps', 1))
    job_id = str(uuid.uuid4())
    jobs[job_id] = {'status': 'queued', 'progress': 0.0}
    thread = threading.Thread(target=run_analysis, args=(job_id, video_url, start_time, end_time, fps))
    thread.start()
    return jsonify({'job_id': job_id})

@app.route('/progress/<job_id>')
def progress(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify({'status': job['status'], 'progress': job.get('progress', 0.0), 'error': job.get('error')})

@app.route('/download/<job_id>/<filetype>')
def download(job_id, filetype):
    job = jobs.get(job_id)
    if not job or job['status'] != 'done':
        return jsonify({'error': 'File not ready'}), 404
    file_map = job['files']
    if filetype not in file_map:
        return jsonify({'error': 'Invalid file type'}), 400
    return send_file(file_map[filetype], as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True) 