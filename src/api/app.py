from flask import Flask, jsonify
from flask_cors import CORS

from .service import run_analysis


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    @app.get("/api/health")
    def health():
        return jsonify({
            "status": "ok",
            "service": "wattsense-api",
        })

    @app.get("/api/demo")
    def demo():
        """
        Temporary endpoint for frontend development.
        Returns mock WattSense analysis.
        """
        return jsonify(run_analysis())

    return app


app = create_app()


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
    )