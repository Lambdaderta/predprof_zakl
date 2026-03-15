from __future__ import annotations

import importlib
import os
import sqlite3
from functools import wraps
from pathlib import Path

from flask import Flask, Response, current_app, jsonify, redirect, render_template, request, send_file, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename


BASE_DIR = Path(__file__).resolve().parent


def hash_password(password: str) -> str:
    return generate_password_hash(password, method="pbkdf2:sha256")


def verify_password(password_hash: str, password: str) -> bool:
    try:
        return check_password_hash(password_hash, password)
    except AttributeError:
        return False


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(current_app.config["DATABASE"])
    conn.row_factory = sqlite3.Row
    return conn


def init_db(app: Flask) -> None:
    with app.app_context():
        conn = get_db()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                role TEXT NOT NULL,
                first_name TEXT,
                last_name TEXT
            )
            """
        )
        cursor = conn.execute("SELECT COUNT(*) FROM users WHERE username = ?", ("admin",))
        if cursor.fetchone()[0] == 0:
            conn.execute(
                "INSERT INTO users (username, password, role, first_name, last_name) VALUES (?, ?, ?, ?, ?)",
                ("admin", hash_password("admin"), "admin", "System", "Admin"),
            )
        else:
            conn.execute(
                "UPDATE users SET password = ? WHERE username = ?",
                (hash_password("admin"), "admin"),
            )
        conn.commit()
        conn.close()


def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return view(*args, **kwargs)

    return wrapped_view


def admin_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if session.get("role") != "admin":
            return redirect(url_for("user_dashboard"))
        return view(*args, **kwargs)

    return wrapped_view


def get_inference_service():
    service = current_app.config.get("INFERENCE_SERVICE")
    if service is not None:
        return service
    service = importlib.import_module("case_web.inference")
    current_app.config["INFERENCE_SERVICE"] = service
    return service


def analytics_error_payload(error: Exception | str) -> dict:
    return {
        "ready": False,
        "error": str(error),
        "model": None,
        "training": None,
        "test": None,
        "plots": {},
        "downloads": {},
    }


def create_app(test_config: dict | None = None) -> Flask:
    app = Flask(__name__, template_folder="templates")
    app.config.from_mapping(
        SECRET_KEY=os.environ.get("SPACE_SECRET_KEY", "space_secret_key_123"),
        DATABASE=str(BASE_DIR / "space_auth.db"),
        MAX_CONTENT_LENGTH=256 * 1024 * 1024,
        INFERENCE_SERVICE=None,
    )

    if test_config:
        app.config.update(test_config)

    init_db(app)

    @app.route("/")
    def index():
        return redirect(url_for("login"))

    @app.route("/health")
    def health():
        return jsonify({"ok": True})

    @app.route("/login", methods=["GET", "POST"])
    def login():
        error = None
        if request.method == "POST":
            username = request.form["username"].strip()
            password = request.form["password"]
            conn = get_db()
            user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
            conn.close()

            if user is None or not verify_password(user["password"], password):
                error = "Неверные данные доступа"
            else:
                session["user_id"] = user["id"]
                session["username"] = user["username"]
                session["role"] = user["role"]
                session["first_name"] = user["first_name"]
                session["last_name"] = user["last_name"]
                if user["role"] == "admin":
                    return redirect(url_for("admin_dashboard"))
                return redirect(url_for("user_dashboard"))

        return render_template("login.html", error=error)

    @app.route("/logout")
    def logout():
        session.clear()
        return redirect(url_for("login"))

    @app.route("/admin", methods=["GET", "POST"])
    @login_required
    @admin_required
    def admin_dashboard():
        error = None
        success = None
        if request.method == "POST":
            username = request.form["username"].strip()
            password = request.form["password"]
            role = request.form["role"]
            first_name = request.form["first_name"].strip()
            last_name = request.form["last_name"].strip()

            conn = get_db()
            try:
                conn.execute(
                    "INSERT INTO users (username, password, role, first_name, last_name) VALUES (?, ?, ?, ?, ?)",
                    (username, hash_password(password), role, first_name, last_name),
                )
                conn.commit()
                success = "Пользователь создан"
            except sqlite3.IntegrityError:
                error = "Пользователь уже существует"
            finally:
                conn.close()

        return render_template("admin.html", error=error, success=success)

    @app.route("/admin/db-dump")
    @login_required
    @admin_required
    def download_db_dump():
        conn = get_db()
        dump_sql = "\n".join(conn.iterdump())
        conn.close()
        return Response(
            dump_sql,
            mimetype="application/sql",
            headers={"Content-Disposition": "attachment; filename=space_auth_dump.sql"},
        )

    @app.route("/user")
    @login_required
    def user_dashboard():
        return render_template("user.html")

    @app.route("/upload", methods=["POST"])
    @login_required
    def upload_data():
        file = request.files.get("file")
        if file is None:
            return jsonify({"ok": False, "error": "Файл не передан"}), 400
        if not file.filename:
            return jsonify({"ok": False, "error": "Не выбран файл"}), 400

        filename = secure_filename(file.filename)
        if not filename.lower().endswith(".npz"):
            return jsonify({"ok": False, "error": "Нужен файл формата .npz"}), 400

        try:
            service = get_inference_service()
            if hasattr(service, "ensure_runtime_dirs"):
                service.ensure_runtime_dirs()
            upload_dir = getattr(service, "UPLOADS_DIR", BASE_DIR / "runtime" / "uploads")
            upload_dir.mkdir(parents=True, exist_ok=True)
            upload_path = upload_dir / filename
            file.save(upload_path)
            result = service.run_test_inference(upload_path)
            return jsonify({"ok": True, "message": "Файл обработан", "result": result})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500

    @app.route("/api/analytics")
    @login_required
    def get_analytics():
        try:
            payload = get_inference_service().build_dashboard_payload()
            return jsonify(payload)
        except Exception as exc:
            return jsonify(analytics_error_payload(exc))

    @app.route("/plots/<path:filename>")
    @login_required
    def get_plot(filename: str):
        try:
            plot_path = get_inference_service().get_plot_path(filename)
            return send_file(plot_path)
        except FileNotFoundError:
            return jsonify({"ok": False, "error": "График не найден"}), 404
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500

    @app.route("/downloads/<path:filename>")
    @login_required
    def download_result(filename: str):
        try:
            result_path = get_inference_service().get_result_path(filename)
            return send_file(result_path, as_attachment=True, download_name=Path(filename).name)
        except FileNotFoundError:
            return jsonify({"ok": False, "error": "Файл не найден"}), 404
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
