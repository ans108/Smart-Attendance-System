"""
admin_auth.py — Simple session-based admin authentication for FaceAttend.

Credentials: ADMIN_USER and ADMIN_PASS environment variables.
Defaults: admin / admin123

Usage:
    from admin_auth import login_required, check_credentials, login_admin, logout_admin

    @app.route('/admin/dashboard')
    @login_required
    def admin_dashboard():
        ...
"""

import os
from functools import wraps

from flask import redirect, session, url_for

# ── Credentials (env vars override defaults) ──────────────────────────────────
ADMIN_USER = os.environ.get("ADMIN_USER", "admin")
ADMIN_PASS = os.environ.get("ADMIN_PASS", "admin123")
SESSION_KEY = "admin_logged_in"


def check_credentials(username: str, password: str) -> bool:
    """Return True if username and password match configured credentials."""
    return username == ADMIN_USER and password == ADMIN_PASS


def login_admin() -> None:
    """Mark the current session as authenticated."""
    session[SESSION_KEY] = True


def logout_admin() -> None:
    """Clear the admin session."""
    session.pop(SESSION_KEY, None)


def is_logged_in() -> bool:
    """Check if the current request session is authenticated as admin."""
    return bool(session.get(SESSION_KEY, False))


def login_required(f):
    """
    Flask route decorator that redirects to /admin if not logged in.
    Applies to all admin-only routes.
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        if not is_logged_in():
            return redirect(url_for("admin_login_page"))
        return f(*args, **kwargs)
    return decorated
