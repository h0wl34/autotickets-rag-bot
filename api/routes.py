from flask import request, jsonify

from src.pipelines.inference_pipeline import InferencePipeline

def register_routes(app):

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"})

    @app.route("/predict", methods=["POST"])
    def predict():
        try:
            data = request.get_json(force=True)
            inputs = data.get("inputs", None)
            if inputs is None:
                return jsonify({"error": "Missing 'inputs' field"}), 400

            # Run inference
            preds = inference_model.predict(inputs)

            return jsonify({"predictions": preds})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
