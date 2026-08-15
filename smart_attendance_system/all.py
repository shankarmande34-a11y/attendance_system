#!/usr/bin/env python3
"""
Smart Face Recognition Attendance System
Single-file production version with Waitress WSGI server.
Usage:
  python attendance.py web        # Start dashboard (WSGI)
  python attendance.py recognize  # Start camera recognition
"""

import os
import sys
import cv2
import numpy as np
import pickle
import logging
from datetime import datetime
from dotenv import load_dotenv
import mysql.connector
from mysql.connector import Error
from scipy.spatial.distance import cosine
import insightface
from flask import Flask, render_template_string, request, send_file
import pandas as pd
from io import BytesIO
from waitress import serve

# ------------------------- Load Environment -------------------------
load_dotenv()

class Config:
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_USER = os.getenv('DB_USER', 'root')
    DB_PASSWORD = os.getenv('DB_PASSWORD', 'shankar@0087')
    DB_NAME = os.getenv('DB_NAME', 'attendance_db')
    SECRET_KEY = os.getenv('SECRET_KEY', 'SECRET_KEY')
    INSIGHTFACE_MODEL = os.getenv('INSIGHTFACE_MODEL', 'buffalo_l')  # <--- CHANGED HERE
    RECOGNITION_THRESHOLD = float(os.getenv('RECOGNITION_THRESHOLD', 0.65))
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    LOG_DIR = os.path.join(BASE_DIR, 'logs')

# ------------------------- Logger -------------------------
def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    fh = logging.FileHandler(os.path.join(Config.LOG_DIR, 'app.log'))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

logger = setup_logger(__name__)

# ------------------------- Helpers -------------------------
def serialize_embedding(emb):
    return pickle.dumps(emb)

def deserialize_embedding(blob):
    return pickle.loads(blob)

# ------------------------- Database -------------------------
class Database:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.connection = None
        self.connect()

    def connect(self):
        try:
            self.connection = mysql.connector.connect(
                host=Config.DB_HOST,
                user=Config.DB_USER,
                password=Config.DB_PASSWORD,
                database=Config.DB_NAME,
                pool_name='mypool',
                pool_size=5
            )
            logger.info("Database connected.")
        except Error as e:
            logger.error(f"DB error: {e}")
            raise

    def execute_query(self, query, params=None, fetch=False):
        cursor = self.connection.cursor(dictionary=True)
        try:
            cursor.execute(query, params)
            if fetch:
                result = cursor.fetchall()
            else:
                self.connection.commit()
                result = cursor.lastrowid
            return result
        except Error as e:
            logger.error(f"Query error: {e}")
            self.connection.rollback()
            raise
        finally:
            cursor.close()

    def close(self):
        if self.connection and self.connection.is_connected():
            self.connection.close()

# ------------------------- Models -------------------------
class Student:
    def __init__(self, student_id=None, name=None, roll_number=None, department=None, year=None):
        self.id = None
        self.student_id = student_id
        self.name = name
        self.roll_number = roll_number
        self.department = department
        self.year = year
        self.db = Database()

    def save(self):
        query = """
            INSERT INTO students (student_id, name, roll_number, department, year)
            VALUES (%s, %s, %s, %s, %s)
        """
        params = (self.student_id, self.name, self.roll_number, self.department, self.year)
        self.id = self.db.execute_query(query, params)
        return self.id

    @classmethod
    def get_all(cls):
        db = Database()
        query = "SELECT * FROM students ORDER BY name"
        return db.execute_query(query, fetch=True)

class FaceEmbedding:
    def __init__(self, student_id=None, embedding=None):
        self.student_id = student_id
        self.embedding = embedding
        self.db = Database()

    def save(self):
        query = """
            INSERT INTO face_embeddings (student_id, embedding)
            VALUES (%s, %s)
            ON DUPLICATE KEY UPDATE embedding = VALUES(embedding)
        """
        blob = serialize_embedding(self.embedding)
        self.db.execute_query(query, (self.student_id, blob))

    @classmethod
    def get_all_embeddings(cls):
        db = Database()
        query = """
            SELECT s.id, s.student_id, s.name, fe.embedding
            FROM students s
            JOIN face_embeddings fe ON s.id = fe.student_id
        """
        rows = db.execute_query(query, fetch=True)
        embeddings = []
        for row in rows:
            emb = deserialize_embedding(row['embedding'])
            embeddings.append({
                'db_id': row['id'],
                'student_id': row['student_id'],
                'name': row['name'],
                'embedding': emb
            })
        return embeddings

class Attendance:
    def __init__(self, student_id=None, date=None, check_in_time=None):
        self.student_id = student_id
        self.date = date or datetime.now().date()
        self.check_in_time = check_in_time or datetime.now().time()
        self.db = Database()

    def mark(self):
        query = """
            INSERT IGNORE INTO attendance (student_id, date, check_in_time)
            VALUES (%s, %s, %s)
        """
        result = self.db.execute_query(query, (self.student_id, self.date, self.check_in_time))
        return result != 0

    @classmethod
    def get_by_date(cls, date_str):
        db = Database()
        query = """
            SELECT a.*, s.name, s.roll_number, s.department
            FROM attendance a
            JOIN students s ON a.student_id = s.id
            WHERE a.date = %s
            ORDER BY a.check_in_time
        """
        return db.execute_query(query, (date_str,), fetch=True)

    @classmethod
    def get_all_dates(cls):
        db = Database()
        query = "SELECT DISTINCT date FROM attendance ORDER BY date DESC"
        return db.execute_query(query, fetch=True)

# ------------------------- Face Service -------------------------
class FaceService:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        # Force CPU if CUDA not available, but InsightFace auto-detects
        self.app = insightface.app.FaceAnalysis(name=Config.INSIGHTFACE_MODEL, root='./')
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        logger.info("InsightFace loaded.")
        self.threshold = Config.RECOGNITION_THRESHOLD
        self.embedding_cache = []
        self.refresh_cache()

    def refresh_cache(self):
        self.embedding_cache = FaceEmbedding.get_all_embeddings()
        logger.info(f"Loaded {len(self.embedding_cache)} embeddings.")

    def extract_embedding(self, face_img):
        faces = self.app.get(face_img)
        if len(faces) == 0:
            return None
        return faces[0].normed_embedding

    def recognize(self, face_img):
        emb = self.extract_embedding(face_img)
        if emb is None:
            return None
        best_match = None
        best_score = -1
        for entry in self.embedding_cache:
            sim = 1 - cosine(emb, entry['embedding'])
            if sim > self.threshold and sim > best_score:
                best_score = sim
                best_match = entry
        if best_match:
            return {
                'db_id': best_match['db_id'],
                'student_id': best_match['student_id'],
                'name': best_match['name'],
                'confidence': best_score
            }
        return None

# ------------------------- Attendance Service -------------------------
class AttendanceService:
    def __init__(self):
        self.face_service = FaceService()

    def process_frame(self, frame):
        faces = self.face_service.app.get(frame)
        for face in faces:
            emb = face.normed_embedding
            recognized = self._recognize_by_embedding(emb)
            if recognized:
                att = Attendance(student_id=recognized['db_id'])
                inserted = att.mark()
                if inserted:
                    logger.info(f"Marked {recognized['name']} at {datetime.now()}")
                bbox = face.bbox.astype(int)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 2)
                label = f"{recognized['name']} ({recognized['confidence']:.2f})"
                cv2.putText(frame, label, (bbox[0], bbox[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            else:
                bbox = face.bbox.astype(int)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,0,255), 2)
                cv2.putText(frame, "Unknown", (bbox[0], bbox[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        return frame

    def _recognize_by_embedding(self, emb):
        best_match = None
        best_score = -1
        for entry in self.face_service.embedding_cache:
            sim = 1 - cosine(emb, entry['embedding'])
            if sim > self.face_service.threshold and sim > best_score:
                best_score = sim
                best_match = entry
        if best_match:
            return {'db_id': best_match['db_id'], 'name': best_match['name'], 'confidence': best_score}
        return None

# ------------------------- Flask App (with templates) -------------------------
def create_app():
    app = Flask(__name__)
    app.secret_key = Config.SECRET_KEY

    # Embedded templates
    BASE = """
    <!DOCTYPE html>
    <html>
    <head><meta charset="UTF-8"><title>Attendance System</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
      <div class="container">
        <a class="navbar-brand" href="/">📸 Smart Attendance</a>
        <ul class="navbar-nav ms-auto">
          <li class="nav-item"><a class="nav-link" href="/students">Students</a></li>
          <li class="nav-item"><a class="nav-link" href="/attendance">Attendance</a></li>
        </ul>
      </div>
    </nav>
    <div class="container mt-4">{% block content %}{% endblock %}</div>
    </body>
    </html>
    """

    INDEX = """
    {% extends "base" %}
    {% block content %}
    <div class="jumbotron">
      <h1 class="display-4">👋 Smart Face Recognition Attendance</h1>
      <p class="lead">Real-time AI-based attendance marking at college entrance.</p>
      <hr class="my-4">
      <p>Use the navigation to view registered students and attendance logs.</p>
    </div>
    {% endblock %}
    """

    STUDENTS = """
    {% extends "base" %}
    {% block content %}
    <h2>📋 Registered Students</h2>
    <table class="table table-striped">
      <thead><tr><th>Student ID</th><th>Name</th><th>Roll No</th><th>Department</th><th>Year</th></tr></thead>
      <tbody>
      {% for s in students %}
      <tr><td>{{ s.student_id }}</td><td>{{ s.name }}</td><td>{{ s.roll_number }}</td><td>{{ s.department }}</td><td>{{ s.year }}</td></tr>
      {% endfor %}
      </tbody>
    </table>
    {% endblock %}
    """

    ATTENDANCE = """
    {% extends "base" %}
    {% block content %}
    <h2>📅 Attendance Log</h2>
    <form method="get" class="row g-3 mb-3">
      <div class="col-auto"><label>Date:</label><input type="date" name="date" value="{{ selected_date }}" class="form-control"></div>
      <div class="col-auto"><button type="submit" class="btn btn-primary">Filter</button>
      <a href="/export?date={{ selected_date }}" class="btn btn-success">📊 Export Excel</a></div>
    </form>
    <table class="table table-bordered">
      <thead><tr><th>Name</th><th>Roll</th><th>Dept</th><th>Check-in Time</th></tr></thead>
      <tbody>
      {% for rec in records %}
      <tr><td>{{ rec.name }}</td><td>{{ rec.roll_number }}</td><td>{{ rec.department }}</td><td>{{ rec.check_in_time }}</td></tr>
      {% else %}
      <tr><td colspan="4">No records for this date.</td></tr>
      {% endfor %}
      </tbody>
    </table>
    {% endblock %}
    """

    @app.route('/')
    def index():
        return render_template_string(INDEX, base=BASE)

    @app.route('/students')
    def students():
        students = Student.get_all()
        return render_template_string(STUDENTS, students=students, base=BASE)

    @app.route('/attendance')
    def attendance():
        date_str = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
        records = Attendance.get_by_date(date_str)
        return render_template_string(ATTENDANCE, records=records, selected_date=date_str, base=BASE)

    @app.route('/export')
    def export_attendance():
        date_str = request.args.get('date')
        if not date_str:
            return "Date parameter required", 400
        records = Attendance.get_by_date(date_str)
        df = pd.DataFrame(records)
        if df.empty:
            return "No records for this date", 404
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Attendance')
        output.seek(0)
        return send_file(output, download_name=f'attendance_{date_str}.xlsx', as_attachment=True)

    return app

# ------------------------- Real-time Recognition -------------------------
def run_recognition():
    att_service = AttendanceService()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Cannot open camera.")
        return
    logger.info("Recognition started. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        annotated = att_service.process_frame(frame)
        cv2.imshow("Attendance System", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# ------------------------- Main Entry -------------------------
if __name__ == '__main__':
    # Ensure database tables exist
    db = Database()
    try:
        db.execute_query("""
            CREATE TABLE IF NOT EXISTS students (
                id INT AUTO_INCREMENT PRIMARY KEY,
                student_id VARCHAR(20) UNIQUE NOT NULL,
                name VARCHAR(100) NOT NULL,
                roll_number VARCHAR(20) UNIQUE NOT NULL,
                department VARCHAR(50),
                year INT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_student_id (student_id),
                INDEX idx_roll (roll_number)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        db.execute_query("""
            CREATE TABLE IF NOT EXISTS face_embeddings (
                id INT AUTO_INCREMENT PRIMARY KEY,
                student_id INT NOT NULL,
                embedding BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE,
                UNIQUE KEY unique_student (student_id)
            ) ENGINE=InnoDB
        """)
        db.execute_query("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INT AUTO_INCREMENT PRIMARY KEY,
                student_id INT NOT NULL,
                date DATE NOT NULL,
                check_in_time TIME NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE,
                UNIQUE KEY unique_attendance (student_id, date),
                INDEX idx_date (date)
            ) ENGINE=InnoDB
        """)
        logger.info("Database tables ensured.")
    except Exception as e:
        logger.error(f"Table creation error: {e}")
        sys.exit(1)

    # Parse command line
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode == 'recognize':
            run_recognition()
        elif mode == 'web':
            app = create_app()
            logger.info("Starting WSGI server with Waitress on port 5000...")
            serve(app, host='0.0.0.0', port=5000)
        else:
            print("Usage: python attendance.py [web|recognize]")
    else:
        print("Please specify mode: web or recognize")