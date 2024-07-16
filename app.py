from flask import Flask, request, Response, redirect, url_for, jsonify, session
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pickle
import base64
import datetime
import os
from pymongo import MongoClient
from flask_pymongo import PyMongo
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, login_user, logout_user, login_required, UserMixin, current_user
from flask_dance.contrib.google import make_google_blueprint, google
from flask_mail import Mail, Message
from flask_jwt_extended import JWTManager, create_access_token, decode_token, jwt_required, get_jwt_identity
from flask_httpauth import HTTPBasicAuth
from dotenv import load_dotenv
from bson.objectid import ObjectId
import uuid
import jwt
from pymongo import TEXT
from flask_cors import CORS

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.config['MONGO_URI'] = os.getenv('MONGO_URI')
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'dindatheanekawati@gmail.com'  
app.config['MAIL_PASSWORD'] = 'mpplplzfsgnazwee'
app.config["JWT_SECRET_KEY"] = os.getenv('JWT_SECRET_KEY')
app.config['MAIL_DEFAULT_SENDER'] = 'dindatheanekawati@gmail.com' 

mongo = PyMongo(app)
bcrypt = Bcrypt(app)
mail = Mail(app)
auth = HTTPBasicAuth()
jwt = JWTManager(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
CORS(app)

google_bp = make_google_blueprint(client_id=os.getenv('GOOGLE_CLIENT_ID'), client_secret=os.getenv('GOOGLE_CLIENT_SECRET'), redirect_to='google_login')
app.register_blueprint(google_bp, url_prefix='/login')

# Load the pre-trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def process_frame(frame, user_data):
    image = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        results = holistic.process(image_rgb)
        image_rgb.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Draw landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(
                                          color=(80, 22, 10), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(
                                          color=(80, 22, 10), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(
                                          color=(121, 22, 76), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))

        userFrame = None

        # Extract pose landmarks
        pose = results.pose_landmarks.landmark
        pose_row = list(np.array(
            [[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

        # Extract face landmarks
        face = results.face_landmarks.landmark
        face_row = list(np.array(
            [[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())

        # Combine pose and face landmarks into one row
        row = pose_row + face_row

        # Predict the body language class
        X = pd.DataFrame([row])
        body_language_class = model.predict(X)[0]
        body_language_prob = model.predict_proba(X)[0]

        # Get coordinates for displaying the class and probability
        coords = tuple(np.multiply(
            np.array(
                (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x,
                 results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y)),
            [640, 480]).astype(int))

        cv2.rectangle(image,
                      (coords[0], coords[1]+5),
                      (coords[0]+len(body_language_class)*20, coords[1]-30),
                      (245, 117, 16), -1)
        cv2.putText(image, body_language_class, coords,
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # status box
        cv2.rectangle(image, (0, 0), (250, 60), (245, 117, 16), -1)

        # Display Class
        cv2.putText(image, 'CLASS', (95, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, body_language_class.split(' ')[0], (90, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Display prob
        cv2.putText(image, 'PROB', (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)], 2)),
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        userFrame = {
            "userID": str(user_data['_id']),
            "namaGerakan": body_language_class,
            "keterangan": "Detected",
            "gender": user_data['gender'],
            "daerah": user_data['daerah'],
            "tanggal": user_data['tanggal']
        }

        # Save the userFrame to MongoDB
        if userFrame:
            mongo.db.deteksi.insert_one(userFrame)

        _, buffer = cv2.imencode('.jpg', image)
        return buffer.tobytes(), userFrame


@app.route('/video_feed', methods=['POST'])
def video_feed():
    frame = base64.b64decode(request.form['frame'])
    email = request.form['email']  # Assume email is sent instead of userID
    user_data = mongo.db.users.find_one({"email": email})

    if not user_data:
        return jsonify({"message": "User not found"}), 404

    user_data['gender'] = request.form['gender']
    user_data['daerah'] = request.form['daerah']
    user_data['tanggal'] = request.form['tanggal']

    processed_frame, _ = process_frame(frame, user_data)
    response = {
        'frame': base64.b64encode(processed_frame).decode('utf-8')
    }
    return jsonify(response)

    
class User(UserMixin):
    def __init__(self, user_data):
        self.id = str(user_data['_id'])
        self.username = user_data['username']
        self.email = user_data['email']
        self.is_verified = user_data.get('is_verified', False)
        self.api_key = user_data.get('api_key')

    @staticmethod
    def create_user(username, email, password=None, google_id=None):
        user = {
            "username": username,
            "email": email,
            "password": bcrypt.generate_password_hash(password).decode('utf-8') if password else None,
            "google_id": google_id,
            "is_verified": False,
            "api_key": str(uuid.uuid4())
        }
        result = mongo.db.users.insert_one(user)
        user['_id'] = str(result.inserted_id)  # Convert ObjectId to string
        return user

    @staticmethod
    def find_by_email(email):
        return mongo.db.users.find_one({"email": email})

    @staticmethod
    def find_by_google_id(google_id):
        return mongo.db.users.find_one({"google_id": google_id})

    @staticmethod
    def verify_password(stored_password, provided_password):
        return bcrypt.check_password_hash(stored_password, provided_password)

    @staticmethod
    def set_verified(user_id):
        mongo.db.users.update_one({'_id': ObjectId(user_id)}, {'$set': {'is_verified': True}})

    def update_password(self, new_password):
        hashed_password = bcrypt.generate_password_hash(new_password).decode('utf-8')
        mongo.db.users.update_one({'_id': ObjectId(self.id)}, {'$set': {'password': hashed_password}})

@login_manager.user_loader
def load_user(user_id):
    user = mongo.db.users.find_one({"_id": ObjectId(user_id)})
    return User(user) if user else None


@auth.verify_password
def verify_password(email, password):
    user_data = User.find_by_email(email)
    if user_data and User.verify_password(user_data['password'], password):
        return User(user_data)
    return None


def verify_api_key(api_key):
    user_data = mongo.db.users.find_one({"api_key": api_key})
    if user_data:
        return User(user_data)
    return None


def decodetoken(jwtToken):
    decode_result = decode_token(jwtToken)
    return decode_result


@app.route('/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')

    if not username or not email or not password:
        return jsonify({"message": "Missing username, email, or password"}), 400

    existing_user = User.find_by_email(email)
    if existing_user:
        if existing_user.get('is_verified', False):
            return jsonify({"message": "Email already registered"}), 400
        else:
            # Resend verification email
            token = create_access_token(identity=str(existing_user['_id']), expires_delta=False)
            msg = Message('Email Verification', recipients=[email])
            msg.body = f'Your verification link is: {token}'
            mail.send(msg)
            return jsonify({"message": "Verification email sent. Please check your inbox."}), 200

    user_data = User.create_user(username=username, email=email, password=password)

    # Send verification email
    token = create_access_token(identity=user_data['_id'], expires_delta=False)
    msg = Message('Email Verification', recipients=[email])
    msg.body = f'Your verification link is: {token}'
    mail.send(msg)

    return jsonify({"message": "User registered successfully. Verification email sent."}), 201


# Define a text index on 'username' and 'email' fields for case-insensitive search
mongo.db.users.create_index([("username", TEXT), ("email", TEXT)], default_language='english')

@app.route('/auth', methods=['GET'])
def detail_user():
    bearer_auth = request.headers.get('Authorization', None)
    if not bearer_auth:
        return {"message": "Authorization header missing"}, 401

    try:
        jwt_token = bearer_auth.split()[1]
        token = decode_token(jwt_token)
        username = token.get('sub')

        if not username:
            return {"message": "Token payload is invalid"}, 401

        user = mongo.db.users.find_one({"_id": ObjectId(username)})
        if not user:
            return {"message": "User not found"}, 404

        # Update is_verified to True
        mongo.db.users.update_one({"_id": user["_id"]}, {"$set": {"is_verified": True}})

        data = {
            'username': user['username'],
            'email': user['email'],
            '_id': str(user['_id'])  # Convert ObjectId to string
        }
    except Exception as e:
        return {
            'message': f'Token is invalid. Please log in again! {str(e)}'
        }, 401

    return jsonify(data), 200

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    print(f'Received login data: {data}')  # Debugging statement
    email = data.get('email')
    password = data.get('password')
    user_data = User.find_by_email(email)
    print(f'User data found: {user_data}')  # Debugging statement

    if user_data and User.verify_password(user_data['password'], password):
        if not user_data.get('is_verified'):
            print('Email not verified')  # Debugging statement
            return jsonify({"message": "Email not verified"}), 403
        user = User(user_data)
        login_user(user)
        access_token = create_access_token(identity=user.id)
        print('Login successful')  # Debugging statement
        return jsonify({'message': 'Login berhasil', 'access_token': access_token}), 200
    print('Invalid credentials')  # Debugging statement
    return jsonify({"message": "Invalid credentials"}), 401

@app.route('/logout', methods=['POST'])
@login_required
def logout():
    logout_user()
    return jsonify({"message": "Logout successful"}), 200

@app.route('/update_password', methods=['POST'])
@login_required
def update_password():
    try:
        data = request.json
        current_password = data.get('Password lama')
        new_password = data.get('Password baru')

        if not current_password or not new_password:
            return jsonify({"message": "Missing current password or new password"}), 400

        user_data = mongo.db.users.find_one({"_id": ObjectId(current_user.id)})
        if not user_data:
            return jsonify({"message": "User not found"}), 404

        if not User.verify_password(user_data['password'], current_password):
            return jsonify({"message": "Current password is incorrect"}), 401

        current_user.update_password(new_password)
        return jsonify({"message": "Password updated successfully"}), 200

    except Exception as e:
        return jsonify({"message": f"An error occurred: {str(e)}"}), 500
    
@app.route('/profile', methods=['GET'])
@login_required
def profile():
    try:
        user_data = mongo.db.users.find_one({"_id": ObjectId(current_user.id)})
        if not user_data:
            return jsonify({"message": "User not found"}), 404

        profile_data = {
            'username': user_data['username'],
            'email': user_data['email'],
            'photo': url_for('static', filename='uploads/' + user_data.get('photo', 'default_profile.jpg'))
        }

        return jsonify(profile_data), 200

    except Exception as e:
        return jsonify({"message": f"An error occurred: {str(e)}"}), 500
    
@app.route('/protected', methods=['GET'])
@jwt_required()
def protected():
    current_user_id = get_jwt_identity()
    user_data = mongo.db.users.find_one({'_id': ObjectId(current_user_id)})
    profile_picture_url = url_for('static', filename='uploads/' + user_data.get('photo', 'default.jpg'))
    return jsonify(
        username=user_data['username'],
        email=user_data['email'],
        profile_picture=profile_picture_url
    ), 200    
    
@app.route('/edit_profile', methods=['POST'])
@login_required
def edit_profile():
    try:
        data = request.form
        username = data.get('username')
        photo = request.files.get('photo')

        if not username:
            return jsonify({"message": "Missing username"}), 400

        user_data = mongo.db.users.find_one({"_id": ObjectId(current_user.id)})
        if not user_data:
            return jsonify({"message": "User not found"}), 404

        update_data = {
            'username': username
        }

        if photo:
            photo_filename = f"{current_user.id}.jpg"
            photo.save(os.path.join('static/uploads', photo_filename))
            update_data['photo'] = photo_filename

        mongo.db.users.update_one({'_id': ObjectId(current_user.id)}, {'$set': update_data})

        return jsonify({"message": "Profile updated successfully"}), 200

    except Exception as e:
        return jsonify({"message": f"An error occurred: {str(e)}"}), 500

@app.route('/change_email', methods=['POST'])
@login_required
def change_email():
    try:
        data = request.json
        new_email = data.get('new_email')

        if not new_email:
            return jsonify({"message": "Missing new email"}), 400

        user_data = mongo.db.users.find_one({"_id": ObjectId(current_user.id)})
        if not user_data:
            return jsonify({"message": "User not found"}), 404

        # Send email confirmation
        token = create_access_token(identity=str(current_user.id), expires_delta=False)
        msg = Message('Email Change Confirmation', recipients=[new_email])
        msg.body = f'Your email change confirmation token is: {token}'
        mail.send(msg)

        return jsonify({"message": "Email change confirmation sent. Please check your inbox."}), 200

    except Exception as e: 
        return jsonify({"message": f"An error occurred: {str(e)}"}), 500

@app.route('/confirm_change_email', methods=['POST'])
def confirm_change_email():
    bearer_auth = request.headers.get('Authorization', None)
    if not bearer_auth:
        return {"message": "Authorization header missing"}, 401

    try:
        jwt_token = bearer_auth.split()[1]
        token = decode_token(jwt_token)
        user_id = token.get('sub')

        if not user_id:
            return {"message": "Token payload is invalid"}, 401

        user = mongo.db.users.find_one({"_id": ObjectId(user_id)})
        if not user:
            return jsonify({"message": "User not found"}), 404

        data = request.json
        new_email = data.get('new_email')

        if not new_email:
            return jsonify({"message": "New email not provided"}), 400

        mongo.db.users.update_one({"_id": ObjectId(user_id)}, {"$set": {"email": new_email}})
        return jsonify({"message": "Email changed successfully"}), 200

    except Exception as e:
        return jsonify({"message": f"An error occurred: {str(e)}"}), 500

@app.route('/forgot_password', methods=['POST'])
def forgot_password():
    data = request.json
    email = data.get('email')

    if not email:
        return jsonify({"message": "Missing email"}), 400

    user_data = User.find_by_email(email)
    if not user_data:
        return jsonify({"message": "User not found"}), 404

    # Kirim email reset password
    token = create_access_token(identity=str(user_data['_id']), expires_delta=False)
    msg = Message('Password Reset', recipients=[email])
    msg.body = f'Your password reset link is: {url_for("update_password", token=token, _external=True)}'
    mail.send(msg)

    return jsonify({"message": "Password reset email sent. Please check your inbox."}), 200

# @app.route('/login/google')
# def google_login():
#     if not google.authorized:
#         return redirect(url_for('google.login'))
#     resp = google.get('/plus/v1/people/me')
#     if not resp.ok:
#         return jsonify({"message": "Google login failed"}), 400
#     google_info = resp.json()
#     google_id = google_info['id']
#     email = google_info['emails'][0]['value']
#     user_data = User.find_by_google_id(google_id)
#     if not user_data:
#         User.create_user(username=google_info['displayName'], email=email, google_id=google_id)
#         user_data = User.find_by_google_id(google_id)
#     user = User(user_data)
#     login_user(user)
#     return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(host='192.168.18.53', port=5000, debug=True)