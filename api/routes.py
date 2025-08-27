from flask import request, jsonify, render_template
import pandas as pd
from datetime import datetime
from src.utils.io_utils import load_yaml
from src.pipelines.inference_pipeline import InferencePipeline

cfg = load_yaml('./configs/inference.yaml')
pipeline = InferencePipeline(cfg)

def register_routes(app):
    
    @app.route("/", methods=["GET"])
    def home():
        
        return render_template("index.html")

    @app.route("/predict", methods=["POST"])
    def predict():
        # Use JSON first, fallback to form
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form
        
        print(data)
        
        df = pd.DataFrame([{
            cfg['UI_TO_DS_COL'][k]: v for k, v in data.items()
        }])
        df['OPEN_TIME_'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        preds = pipeline.run(df)
        return jsonify(preds)
    
    @app.route("/meta", methods=["GET"])
    def get_meta():
        try:
            # Only include fields that actually have encoders
            valid_classes = {
                ui_name: pipeline.fe.cat_encoders[ds_col].categories_[0].tolist()
                for ui_name, ds_col in cfg['UI_TO_DS_COL'].items()
                if ds_col in pipeline.fe.cat_encoders
            }
            return jsonify(valid_classes)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
