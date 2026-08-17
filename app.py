import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# Initialize MediaPipe Hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# --- GESTURE RECOGNITION LOGIC ---
def classify_gesture(frame):
    """
    Processes a frame to find a hand and classify it into Rock, Paper, or Scissors.
    Returns: (processed_frame, gesture_name, alert_message)
    """
    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    gesture = "Unknown"
    alert = "⚠️ No hand detected! Please show your hand to the camera."
    
    if results.multi_hand_landmarks:
        alert = ""  # Hand found, clear default alert
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw skeleton overlays on the frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get tracking coordinates for finger tips and joints
            # Key points: Thumb(4), Index(8), Middle(12), Ring(16), Pinky(20)
            landmarks = hand_landmarks.landmark
            
            # Check if fingers are extended (Tip is higher up/y-axis lesser than PIP joint)
            fingers = []
            
            # Thumb (Horizontal check depending on handedness, simplified approach)
            if landmarks[4].x < landmarks[3].x:
                fingers.append(1)
            else:
                fingers.append(0)
                
            # 4 Fingers (Vertical check)
            finger_tips = [8, 12, 16, 20]
            finger_pips = [6, 10, 14, 18]
            
            for tip, pip in zip(finger_tips, finger_pips):
                if landmarks[tip].y < landmarks[pip].y:
                    fingers.append(1) # Finger is open
                else:
                    fingers.append(0) # Finger is closed

            total_fingers = sum(fingers)
            
            # Map finger counts to Rock, Paper, Scissors rules
            if total_fingers == 0 or total_fingers == 1:
                gesture = "Rock"
            elif total_fingers == 2 and fingers[1] == 1 and fingers[2] == 1:
                gesture = "Scissors"
            elif total_fingers == 5:
                gesture = "Paper"
            else:
                gesture = "Invalid"
                alert = "⚠️ Invalid gesture! Please clearly show Rock, Paper, or Scissors."
                
    return frame, gesture, alert

# --- STREAMLIT UI SETUP ---
st.title("🤖 Two-Round RPS Vision Game")
st.write("Play two consecutive rounds against yourself or a friend, analyzed via computer vision!")

# Initialize session state variables to track game progression
if "game_state" not in st.session_state:
    st.session_state.game_state = "Round 1"  # Options: Round 1, Round 2, Finished
if "round_1_choice" not in st.session_state:
    st.session_state.round_1_choice = None
if "round_2_choice" not in st.session_state:
    st.session_state.round_2_choice = None

# Display Current Game Progress Status
st.subheader(f"Status: {st.session_state.game_state}")

# STAGE 1: Webcam Input Processing
if st.session_state.game_state in ["Round 1", "Round 2"]:
    
    # Configure WebRTC to bypass firewalls on mobile connections (STUN Server)
    rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:://google.com"]}]})
    
    ctx = webrtc_streamer(
        key="rps-stream",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_config,
        video_html_attrs={"playsinline": True, "muted": True, "controls": False}, # Crucial for mobile execution
        media_stream_constraints={"video": True, "audio": False},
    )

    # If camera stream is successfully active
    if ctx.video_receiver:
        try:
            # Read latest video frame sent by the client phone browser
            frame = ctx.video_receiver.get_frame()
            img = frame.to_ndarray(format="bgr24")
            
            # Analyze gesture
            annotated_img, current_gesture, alert_msg = classify_gesture(img)
            
            # Show live camera analysis feedback right on Streamlit UI dashboard
            st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), channels="RGB")
            
            if alert_msg:
                st.error(alert_msg)
            else:
                st.success(f"Detected Gesture: **{current_gesture}**")
                
            # Submit action button 
            if st.button("🔒 Lock Input Gesture", use_container_width=True):
                if current_gesture in ["Unknown", "Invalid"]:
                    st.warning("Cannot lock input! Fix your hand position first.")
                else:
                    if st.session_state.game_state == "Round 1":
                        st.session_state.round_1_choice = current_gesture
                        st.session_state.game_state = "Round 2"
                        st.rerun()
                    elif st.session_state.game_state == "Round 2":
                        st.session_state.round_2_choice = current_gesture
                        st.session_state.game_state = "Finished"
                        st.rerun()
                        
        except Exception as e:
            st.info("Awaiting video frames from camera...")

# STAGE 2: Game Logic Results Breakdown
if st.session_state.game_state == "Finished":
    st.balloons()
    st.markdown("### 🏆 Game Summary Results")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Round 1 Choice", value=st.session_state.round_1_choice)
    with col2:
        st.metric(label="Round 2 Choice", value=st.session_state.round_2_choice)
        
    # Standard evaluation matrix logic
    r1 = st.session_state.round_1_choice
    r2 = st.session_state.round_2_choice
    
    if r1 == r2:
        result_text = "It's an overall Tie Match! 🤝"
    elif (r1 == "Rock" and r2 == "Scissors") or (r1 == "Scissors" and r2 == "Paper") or (r1 == "Paper" and r2 == "Rock"):
        result_text = "🏆 Round 1 Choice Beats Round 2 Choice!"
    else:
        result_text = "🏆 Round 2 Choice Beats Round 1 Choice!"
        
    st.info(f"**Final Verdict:** {result_text}")
    
    if st.button("🔄 Play Again", use_container_width=True):
        st.session_state.game_state = "Round 1"
        st.session_state.round_1_choice = None
        st.session_state.round_2_choice = None
        st.rerun()