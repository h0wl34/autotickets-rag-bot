from flask import Flask
from api.routes import register_routes

def create_app():
    app = Flask()
    register_routes(app)
    return app

app = create_app()
