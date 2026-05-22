import hmac
import os
import secrets
from flask import Blueprint, render_template, request, redirect, url_for, session, flash, current_app
from werkzeug.security import check_password_hash

auth_bp = Blueprint('auth', __name__)


def _configured_admin_username():
    return os.getenv("APP_ADMIN_USERNAME", "admin")


def _validate_login_credentials(username, password):
    configured_username = _configured_admin_username()
    configured_password_hash = os.getenv("APP_ADMIN_PASSWORD_HASH")
    configured_password = os.getenv("APP_ADMIN_PASSWORD")

    if not username or not password:
        return False

    if not hmac.compare_digest(str(username), str(configured_username)):
        return False

    if configured_password_hash:
        return check_password_hash(configured_password_hash, password)

    if configured_password is not None:
        return hmac.compare_digest(str(password), str(configured_password))

    current_app.logger.warning(
        "APP_ADMIN_PASSWORD / APP_ADMIN_PASSWORD_HASH belum diset. "
        "Sistem masih memakai credential fallback lama. Segera pindah ke environment variable."
    )
    return hmac.compare_digest(str(password), "admin")

@auth_bp.route('/')
def index():
    return redirect(url_for('auth.login'))

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('user') and session.get('session_token'):
        return redirect(url_for('dashboard.dashboard'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        if _validate_login_credentials(username, password):
            session.clear()
            session.permanent = True
            session['user'] = username
            session['session_token'] = secrets.token_urlsafe(32)
            return redirect(url_for('dashboard.dashboard'))
        else:
            flash('Username atau password salah!')
    return render_template('login.html')

@auth_bp.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('auth.login'))
