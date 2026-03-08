from flask import Flask, request, jsonify
import librosa
import numpy as np
import io

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400
    
    file = request.files['file']
    # تحميل ومعالجة الصوت
    y, sr = librosa.load(io.BytesIO(file.read()))
    
    # استخراج القيم الأساسية
    rms = np.mean(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
    
    return jsonify({
        "RMS_Mean": float(rms),
        "ZCR_Mean": float(zcr),
        "MFCCs1": float(mfcc[0]),
        "MFCCs2": float(mfcc[1])
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)