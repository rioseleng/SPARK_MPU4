"""Two-Round Rock-Paper-Scissors Vision Game.

Live webcam game: Streamlit UI + streamlit-webrtc video pipeline,
MediaPipe HandLandmarker for hand gesture detection, OpenCV for the
skeleton overlay and on-frame drawing.

Deploy on Streamlit Community Cloud
-----------------------------------
1. Commit app.py, hand_landmarker.task, requirements.txt and packages.txt.
2. In Streamlit Cloud, create a new app from the GitHub repo and choose a
   Python version >= 3.10 (3.11 or 3.12 recommended).
3. packages.txt installs the apt libraries (libGL, glib, libgomp, ...) that
   OpenCV/MediaPipe need on the Linux runtime, so no system-dependency
   errors occur.
4. If hand_landmarker.task is not committed to the repo, the app downloads
   it automatically on first run, so the repo works either way.
"""

import math
import os
import threading
import time
import urllib.request

import av
import cv2
import mediapipe as mp
import streamlit as st
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

# STUN server so the WebRTC connection works on mobile networks / NATs.
RTC_CONFIGURATION = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

# Low-ish resolution + front camera: fast on mobile and on Cloud CPUs.
MEDIA_STREAM_CONSTRAINTS = {
    "video": {"width": {"ideal": 640}, "height": {"ideal": 480}, "facingMode": "user"},
    "audio": False,
}

MAX_PROCESS_WIDTH = 640  # frames are downscaled to this width before inference
EXTEND_ANGLE = 155.0     # min angle (deg) at the PIP joint for a finger to count as extended

BEATS = {"rock": "scissors", "scissors": "paper", "paper": "rock"}
VALID_GESTURES = frozenset(BEATS)
GESTURE_LABEL = {"rock": "Rock", "paper": "Paper", "scissors": "Scissors"}
GESTURE_EMOJI = {"rock": "✊", "paper": "✋", "scissors": "✌️"}

HAND_CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS

WRIST, THUMB_IP, THUMB_TIP = 0, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_MCP, RING_PIP, RING_TIP = 13, 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20


def angle_at(a, b, c):
    """Angle in degrees at vertex b formed by segments b->a and b->c."""
    v1 = (a.x - b.x, a.y - b.y)
    v2 = (c.x - b.x, c.y - b.y)
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    n1 = math.hypot(v1[0], v1[1])
    n2 = math.hypot(v2[0], v2[1])
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / (n1 * n2)))))


def classify_gesture(landmarks, handedness):
    """Classify the 21 hand landmarks into rock / paper / scissors (or None)."""
    index_open = angle_at(landmarks[INDEX_MCP], landmarks[INDEX_PIP], landmarks[INDEX_TIP]) > EXTEND_ANGLE
    middle_open = angle_at(landmarks[MIDDLE_MCP], landmarks[MIDDLE_PIP], landmarks[MIDDLE_TIP]) > EXTEND_ANGLE
    ring_open = angle_at(landmarks[RING_MCP], landmarks[RING_PIP], landmarks[RING_TIP]) > EXTEND_ANGLE
    pinky_open = angle_at(landmarks[PINKY_MCP], landmarks[PINKY_PIP], landmarks[PINKY_TIP]) > EXTEND_ANGLE

    if handedness == "Left":
        thumb_open = landmarks[THUMB_TIP].x > landmarks[THUMB_IP].x
    else:
        thumb_open = landmarks[THUMB_TIP].x < landmarks[THUMB_IP].x

    if not (index_open or middle_open or ring_open or pinky_open or thumb_open):
        return "rock"
    if index_open and middle_open and ring_open and pinky_open and thumb_open:
        return "paper"
    if index_open and middle_open and not ring_open and not pinky_open and not thumb_open:
        return "scissors"
    return None


def draw_skeleton(img, landmarks):
    """Overlay the hand skeleton (connections + joints) on the frame."""
    h, w = img.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, pts[a], pts[b], (0, 255, 150), 2, cv2.LINE_AA)
    for px, py in pts:
        cv2.circle(img, (px, py), 4, (0, 200, 255), -1, cv2.LINE_AA)


def draw_banner(img, gesture, hand_present):
    """Draw the current detection status at the top of the frame."""
    if gesture:
        text, color = f"Gesture: {GESTURE_LABEL[gesture]}", (0, 200, 0)
    elif hand_present:
        text, color = "Show Rock / Paper / Scissors", (0, 165, 255)
    else:
        text, color = "No hand detected", (120, 120, 120)
    cv2.rectangle(img, (0, 0), (img.shape[1], 52), (20, 20, 20), -1)
    cv2.putText(img, text, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)


def ensure_model():
    """Return the local model path, downloading it on first use if needed."""
    if os.path.exists(MODEL_PATH):
        return MODEL_PATH
    tmp = f"{MODEL_PATH}.part"
    urllib.request.urlretrieve(MODEL_URL, tmp)
    os.replace(tmp, MODEL_PATH)
    return MODEL_PATH


def create_landmarker():
    base_options = python.BaseOptions(
        model_asset_path=ensure_model(), delegate=python.BaseOptions.Delegate.CPU
    )
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


class RPSVideoProcessor(VideoProcessorBase):
    """Per-frame video processor: landmarks -> gesture -> annotated frame."""

    def __init__(self):
        self._landmarker = create_landmarker()
        self._lock = threading.Lock()
        self._last_ts_ms = 0
        self.latest_gesture = None
        self.hand_present = False

    def _next_ts_ms(self):
        ts = int(time.time() * 1000)
        if ts <= self._last_ts_ms:
            ts = self._last_ts_ms + 1
        self._last_ts_ms = ts
        return ts

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # mirror view, like a selfie camera

        h, w = img.shape[:2]
        scale = min(1.0, MAX_PROCESS_WIDTH / w)
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)))

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        with self._lock:
            result = self._landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), self._next_ts_ms()
            )

        gesture = None
        hand_present = bool(result.hand_landmarks)
        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            handedness = result.handedness[0][0].category_name
            gesture = classify_gesture(landmarks, handedness)
            draw_skeleton(img, landmarks)

        self.latest_gesture = gesture
        self.hand_present = hand_present
        draw_banner(img, gesture, hand_present)

        if scale < 1.0:
            img = cv2.resize(img, (w, h))
        return av.VideoFrame.from_ndarray(img, format="bgr24")


def lock_round(attr, gesture):
    st.session_state[attr] = gesture


def reset_game():
    st.session_state.round1 = None
    st.session_state.round2 = None


def get_winner(r1, r2):
    if r1 == r2:
        return "tie"
    return "round1" if BEATS[r1] == r2 else "round2"


def main():
    st.set_page_config(page_title="Two-Round Rock-Paper-Scissors", page_icon="✌️", layout="centered")
    st.title("✌️ Two-Round Rock–Paper–Scissors Vision Game")
    st.caption("Show your hand to the camera and lock in your throw for each round.")

    if "round1" not in st.session_state:
        reset_game()

    if not os.path.exists(MODEL_PATH):
        with st.status("Downloading the MediaPipe hand-landmark model..."):
            ensure_model()

    with st.expander("How to play"):
        st.markdown(
            "- **Rock** - closed fist, all fingers (and thumb) tucked in\n"
            "- **Paper** - all five fingers spread open\n"
            "- **Scissors** - strictly only the index and middle fingers extended, thumb tucked\n"
            "- Lock **Round 1** first, then **Round 2**. Once both rounds are locked, the winner is announced."
        )

    col_camera, col_game = st.columns([7, 5], gap="large")

    with col_camera:
        st.subheader("Live Camera")
        ctx = webrtc_streamer(
            key="rps-vision",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints=MEDIA_STREAM_CONSTRAINTS,
            video_processor_factory=RPSVideoProcessor,
            async_processing=True,
            sendback_audio=False,
        )
        current = ctx.video_processor.latest_gesture if ctx.video_processor else None
        if current:
            st.metric("Current gesture", f"{GESTURE_EMOJI[current]} {GESTURE_LABEL[current]}")
        else:
            st.metric("Current gesture", "-")

    with col_game:
        st.subheader("Rounds")
        round1 = st.session_state.round1
        round2 = st.session_state.round2
        can_lock = current in VALID_GESTURES

        if round1 is None:
            st.markdown("### Round 1 :hourglass:")
            st.caption("Not locked yet")
            st.button(
                "Lock Round 1",
                disabled=not can_lock,
                on_click=lock_round,
                args=("round1", current),
                key="lock_round_1",
            )
        else:
            st.markdown(f"### Round 1 {GESTURE_EMOJI[round1]}")
            st.caption(f"Locked: **{GESTURE_LABEL[round1]}**")

        if round1 is None:
            st.info("Lock Round 1 to enable Round 2.")
        elif round2 is None:
            st.markdown("### Round 2 :hourglass:")
            st.caption("Not locked yet")
            st.button(
                "Lock Round 2",
                disabled=not can_lock,
                on_click=lock_round,
                args=("round2", current),
                key="lock_round_2",
            )
        else:
            st.markdown(f"### Round 2 {GESTURE_EMOJI[round2]}")
            st.caption(f"Locked: **{GESTURE_LABEL[round2]}**")

        st.caption("Lock buttons activate when the camera detects a valid rock/paper/scissors gesture.")

        if round1 is not None and round2 is not None:
            st.divider()
            st.subheader("Result")
            winner = get_winner(round1, round2)
            if winner == "tie":
                st.success(
                    f"It's a tie! Both rounds threw {GESTURE_EMOJI[round1]} **{GESTURE_LABEL[round1]}**."
                )
            else:
                winning = round1 if winner == "round1" else round2
                st.success(
                    f"**Round {winner[-1]} wins!** {GESTURE_EMOJI[winning]} **{GESTURE_LABEL[winning]}** "
                    f"beats the other round's throw."
                )
            st.button("Play Again", type="primary", on_click=reset_game)


if __name__ == "__main__":
    main()
