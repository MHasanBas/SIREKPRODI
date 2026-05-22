import os
import secrets
from datetime import timedelta
from flask import Flask
from flask import request, redirect, url_for, session, jsonify

# Import blueprints
from routes.auth import auth_bp
from routes.upload import upload_bp
from routes.dashboard import dashboard_bp
from routes.model import model_bp

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY') or secrets.token_hex(32)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = os.getenv('SESSION_COOKIE_SECURE', '0') == '1'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)

# Buat folder yang diperlukan
for folder in ['uploads', 'models/model_utama']:
    os.makedirs(folder, exist_ok=True)

# Register Blueprints
app.register_blueprint(auth_bp)
app.register_blueprint(upload_bp)
app.register_blueprint(dashboard_bp)
app.register_blueprint(model_bp)


@app.before_request
def enforce_authentication():
    public_endpoints = {
        'auth.index',
        'auth.login',
        'auth.logout',
        'static',
    }

    endpoint = request.endpoint or ''
    if endpoint in public_endpoints or endpoint.startswith('static'):
        return None

    is_authenticated = bool(session.get('user') and session.get('session_token'))
    if is_authenticated:
        return None

    wants_json = (
        request.path.startswith('/api/')
        or request.headers.get('X-Requested-With') == 'XMLHttpRequest'
        or request.accept_mimetypes['application/json'] >= request.accept_mimetypes['text/html']
    )
    if wants_json:
        return jsonify({"success": False, "error": "Unauthorized"}), 401
    return redirect(url_for('auth.login'))

if __name__ == '__main__':
    app.run(debug=os.getenv('FLASK_DEBUG', '0') == '1')
