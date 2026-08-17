import streamlit as st
import cv2
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Two-Round RPS Vision Game",
    page_icon="✋",
    layout="centered"
)

st.title("🤖 Two-Round RPS Vision Game")
st.write(
    "Play two consecutive rounds against yourself or a friend, "
    "analyzed using computer vision."
)


# ============================================================
# MEDIAPIPE INITIALIZATION
# ============================================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)


# ============================================================
# GESTURE RECOGNITION
# ============================================================

def classify_gesture(frame):
    """
    Detects a hand and classifies it as:
    Rock, Paper, Scissors, Invalid, or Unknown.

    Returns:
        processed_frame
        gesture
        alert_message
    """

    # Convert BGR -> RGB for MediaPipe
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb_frame)

    gesture = "Unknown"
    alert = "⚠️ No hand detected! Please show your hand to the camera."

    if results.multi_hand_landmarks:

        alert = ""

        for hand_landmarks in results.multi_hand_landmarks:

            # Draw hand landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            landmarks = hand_landmarks.landmark

            # ------------------------------------------------
            # Detect fingers
            # ------------------------------------------------

            fingers = []

            # Thumb
            if landmarks[4].x < landmarks[3].x:
                fingers.append(1)
            else:
                fingers.append(0)

            # Index, Middle, Ring, Pinky
            finger_tips = [8, 12, 16, 20]
            finger_pips = [6, 10, 14, 18]

            for tip, pip in zip(finger_tips, finger_pips):

                if landmarks[tip].y < landmarks[pip].y:
                    fingers.append(1)
                else:
                    fingers.append(0)

            total_fingers = sum(fingers)

            # ------------------------------------------------
            # Gesture classification
            # ------------------------------------------------

            if total_fingers == 0 or total_fingers == 1:

                gesture = "Rock"

            elif (
                total_fingers == 2
                and fingers[1] == 1
                and fingers[2] == 1
            ):

                gesture = "Scissors"

            elif total_fingers == 5:

                gesture = "Paper"

            else:

                gesture = "Invalid"

                alert = (
                    "⚠️ Invalid gesture! "
                    "Please clearly show Rock, Paper, or Scissors."
                )

    return frame, gesture, alert


# ============================================================
# WEBRTC VIDEO PROCESSOR
# ============================================================

class VideoProcessor:

    def __init__(self):
        self.gesture = "Unknown"
        self.alert = (
            "⚠️ No hand detected! "
            "Please show your hand to the camera."
        )

    def recv(self, frame):

        # Convert WebRTC frame to OpenCV format
        img = frame.to_ndarray(format="bgr24")

        # Analyze gesture
        processed_img, gesture, alert = classify_gesture(img)

        # Save latest detection
        self.gesture = gesture
        self.alert = alert

        # Return processed frame
        return av.VideoFrame.from_ndarray(
            processed_img,
            format="bgr24"
        )


# ============================================================
# SESSION STATE
# ============================================================

if "game_state" not in st.session_state:
    st.session_state.game_state = "Round 1"

if "round_1_choice" not in st.session_state:
    st.session_state.round_1_choice = None

if "round_2_choice" not in st.session_state:
    st.session_state.round_2_choice = None


# ============================================================
# GAME STATUS
# ============================================================

st.subheader(
    f"Status: {st.session_state.game_state}"
)


# ============================================================
# ROUND 1 / ROUND 2
# ============================================================

if st.session_state.game_state in ["Round 1", "Round 2"]:

    st.info(
        f"Show your hand for {st.session_state.game_state}."
    )

    # --------------------------------------------------------
    # WebRTC configuration
    # --------------------------------------------------------

    rtc_config = RTCConfiguration(
        {
            "iceServers": [
                {
                    "urls": [
                        "stun:stun.l.google.com:19302"
                    ]
                }
            ]
        }
    )

    # --------------------------------------------------------
    # Start WebRTC
    # --------------------------------------------------------

    ctx = webrtc_streamer(
        key="rps-stream",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_config,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        video_html_attrs={
            "playsinline": True,
            "muted": True,
            "controls": False
        }
    )

    # --------------------------------------------------------
    # Display current detection
    # --------------------------------------------------------

    if ctx.video_processor:

        current_gesture = ctx.video_processor.gesture
        alert_msg = ctx.video_processor.alert

        if current_gesture in [
            "Rock",
            "Paper",
            "Scissors"
        ]:

            st.success(
                f"Detected Gesture: **{current_gesture}**"
            )

        elif current_gesture == "Invalid":

            st.warning(alert_msg)

        else:

            st.error(alert_msg)

        # ----------------------------------------------------
        # Lock gesture
        # ----------------------------------------------------

        if st.button(
            "🔒 Lock Input Gesture",
            use_container_width=True
        ):

            if current_gesture in [
                "Unknown",
                "Invalid"
            ]:

                st.warning(
                    "Cannot lock input! "
                    "Please show a valid gesture."
                )

            else:

                if st.session_state.game_state == "Round 1":

                    st.session_state.round_1_choice = (
                        current_gesture
                    )

                    st.session_state.game_state = "Round 2"

                    st.rerun()

                elif st.session_state.game_state == "Round 2":

                    st.session_state.round_2_choice = (
                        current_gesture
                    )

                    st.session_state.game_state = "Finished"

                    st.rerun()


# ============================================================
# FINAL RESULTS
# ============================================================

if st.session_state.game_state == "Finished":

    st.balloons()

    st.markdown("### 🏆 Game Summary Results")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            label="Round 1 Choice",
            value=st.session_state.round_1_choice
        )

    with col2:

        st.metric(
            label="Round 2 Choice",
            value=st.session_state.round_2_choice
        )

    # --------------------------------------------------------
    # Determine winner
    # --------------------------------------------------------

    r1 = st.session_state.round_1_choice
    r2 = st.session_state.round_2_choice

    if r1 == r2:

        result_text = (
            "It's an overall Tie Match! 🤝"
        )

    elif (
        (r1 == "Rock" and r2 == "Scissors")
        or
        (r1 == "Scissors" and r2 == "Paper")
        or
        (r1 == "Paper" and r2 == "Rock")
    ):

        result_text = (
            "🏆 Round 1 Choice Beats Round 2 Choice!"
        )

    else:

        result_text = (
            "🏆 Round 2 Choice Beats Round 1 Choice!"
        )

    st.info(
        f"**Final Verdict:** {result_text}"
    )

    # --------------------------------------------------------
    # Play again
    # --------------------------------------------------------

    if st.button(
        "🔄 Play Again",
        use_container_width=True
    ):

        st.session_state.game_state = "Round 1"
        st.session_state.round_1_choice = None
        st.session_state.round_2_choice = None

        st.rerun()