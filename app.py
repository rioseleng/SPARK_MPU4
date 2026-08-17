import streamlit as st
import cv2
import mediapipe as mp
import av

from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    RTCConfiguration
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Two-Round RPS Vision Game",
    page_icon="✋",
    layout="centered"
)


# ============================================================
# TITLE
# ============================================================

st.title("🤖 Two-Round RPS Vision Game")

st.write(
    "Play two consecutive rounds of Rock-Paper-Scissors "
    "using computer vision."
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
# GESTURE CLASSIFICATION
# ============================================================

def classify_gesture(frame):

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Process frame with MediaPipe
    results = hands.process(rgb_frame)

    gesture = "Unknown"

    alert = (
        "⚠️ No hand detected! "
        "Please show your hand to the camera."
    )

    # ========================================================
    # HAND DETECTED
    # ========================================================

    if results.multi_hand_landmarks:

        alert = ""

        for hand_landmarks in results.multi_hand_landmarks:

            # Draw hand skeleton
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            landmarks = hand_landmarks.landmark

            # =================================================
            # FINGER DETECTION
            # =================================================

            fingers = []

            # -------------------------------------------------
            # Thumb
            # -------------------------------------------------

            if landmarks[4].x < landmarks[3].x:
                fingers.append(1)
            else:
                fingers.append(0)

            # -------------------------------------------------
            # Other four fingers
            # -------------------------------------------------

            finger_tips = [8, 12, 16, 20]
            finger_pips = [6, 10, 14, 18]

            for tip, pip in zip(
                finger_tips,
                finger_pips
            ):

                if landmarks[tip].y < landmarks[pip].y:
                    fingers.append(1)
                else:
                    fingers.append(0)

            total_fingers = sum(fingers)

            # =================================================
            # ROCK
            # =================================================

            if total_fingers == 0 or total_fingers == 1:

                gesture = "Rock"

            # =================================================
            # SCISSORS
            # =================================================

            elif (
                total_fingers == 2
                and fingers[1] == 1
                and fingers[2] == 1
            ):

                gesture = "Scissors"

            # =================================================
            # PAPER
            # =================================================

            elif total_fingers == 5:

                gesture = "Paper"

            # =================================================
            # INVALID
            # =================================================

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

        # Convert WebRTC frame to OpenCV image
        img = frame.to_ndarray(
            format="bgr24"
        )

        # Analyze gesture
        processed_img, gesture, alert = (
            classify_gesture(img)
        )

        # Save latest detection
        self.gesture = gesture
        self.alert = alert

        # Return processed video frame
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
# GAME ROUNDS
# ============================================================

if st.session_state.game_state in [
    "Round 1",
    "Round 2"
]:

    st.info(
        f"Show your hand for "
        f"**{st.session_state.game_state}**."
    )

    # ========================================================
    # WEBRTC CONFIGURATION
    # ========================================================

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

    # ========================================================
    # START CAMERA
    # ========================================================

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

    # ========================================================
    # SHOW DETECTION
    # ========================================================

    if ctx.video_processor:

        current_gesture = (
            ctx.video_processor.gesture
        )

        alert_msg = (
            ctx.video_processor.alert
        )

        # ----------------------------------------------------
        # Valid gesture
        # ----------------------------------------------------

        if current_gesture in [
            "Rock",
            "Paper",
            "Scissors"
        ]:

            st.success(
                f"Detected Gesture: "
                f"**{current_gesture}**"
            )

        # ----------------------------------------------------
        # Invalid gesture
        # ----------------------------------------------------

        elif current_gesture == "Invalid":

            st.warning(alert_msg)

        # ----------------------------------------------------
        # No hand
        # ----------------------------------------------------

        else:

            st.error(alert_msg)

        # ====================================================
        # LOCK GESTURE
        # ====================================================

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

                # --------------------------------------------
                # ROUND 1
                # --------------------------------------------

                if (
                    st.session_state.game_state
                    == "Round 1"
                ):

                    st.session_state.round_1_choice = (
                        current_gesture
                    )

                    st.session_state.game_state = (
                        "Round 2"
                    )

                    st.rerun()

                # --------------------------------------------
                # ROUND 2
                # --------------------------------------------

                elif (
                    st.session_state.game_state
                    == "Round 2"
                ):

                    st.session_state.round_2_choice = (
                        current_gesture
                    )

                    st.session_state.game_state = (
                        "Finished"
                    )

                    st.rerun()


# ============================================================
# FINAL RESULTS
# ============================================================

if st.session_state.game_state == "Finished":

    st.balloons()

    st.markdown(
        "### 🏆 Game Summary Results"
    )

    col1, col2 = st.columns(2)

    # ========================================================
    # ROUND 1 RESULT
    # ========================================================

    with col1:

        st.metric(
            label="Round 1 Choice",
            value=st.session_state.round_1_choice
        )

    # ========================================================
    # ROUND 2 RESULT
    # ========================================================

    with col2:

        st.metric(
            label="Round 2 Choice",
            value=st.session_state.round_2_choice
        )

    # ========================================================
    # DETERMINE WINNER
    # ========================================================

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
            "🏆 Round 1 Choice Beats "
            "Round 2 Choice!"
        )

    else:

        result_text = (
            "🏆 Round 2 Choice Beats "
            "Round 1 Choice!"
        )

    st.info(
        f"**Final Verdict:** {result_text}"
    )

    # ========================================================
    # PLAY AGAIN
    # ========================================================

    if st.button(
        "🔄 Play Again",
        use_container_width=True
    ):

        st.session_state.game_state = "Round 1"

        st.session_state.round_1_choice = None

        st.session_state.round_2_choice = None

        st.rerun()