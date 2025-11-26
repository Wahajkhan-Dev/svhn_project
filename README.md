# SVHN — Model Evaluation & Interpretability


- train.py — download TFDS svhn_cropped, train/save model & preds
- evaluate.py — confusion matrix, mis_idx
- gradcam.py — generate Grad-CAM overlays
- dashboard.py — Plotly Dash to browse examples, misclassifications, Grad-CAMs
- utils.py — helpers (preprocess, noise, blur)

1. pip install -r requirements.txt
2. python train.py --epochs 5 --batch 64 --save-dir outputs
3. python evaluate.py --out outputs
4. python gradcam.py --model outputs/model.h5 --out outputs/gradcam
5. python dashboard.py --out outputs --port 8050
