# GestureCanvas AI
# main.py

import cv2
import mediapipe as mp
import numpy as np
import math
import collections # Added for deque
import time # For generating unique filenames
import os # For environment variables
from dotenv import load_dotenv # For loading .env file
from langchain_google_genai import ChatGoogleGenerativeAI # For Gemini
from langchain_core.messages import HumanMessage # For formatting LLM input

# Helper function to calculate distance between two landmarks
def calculate_distance(lm1, lm2):
    return math.sqrt((lm1.x - lm2.x)**2 + (lm1.y - lm2.y)**2 + (lm1.z - lm2.z)**2)

# --- LLM Integration ---
llm = None

def init_llm():
    global llm
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("GOOGLE_API_KEY not found in environment variables or .env file.")
        print("LLM suggestions will be disabled.")
        return False
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key, convert_system_message_to_human=True)
        print("LLM (Gemini Pro) initialized successfully.")
        return True
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        llm = None
        return False

def get_llm_suggestion(prompt_text):
    if not llm:
        return "LLM not initialized. Cannot get suggestions."
    try:
        message = HumanMessage(content=prompt_text)
        response = llm.invoke([message])
        return response.content
    except Exception as e:
        return f"Error getting suggestion from LLM: {e}"
# --- End LLM Integration ---

def main():
    print("GestureCanvas AI starting...")
    llm_initialized = init_llm()

    mp_hands = mp.solutions.hands
    # Update max_num_hands to 2 for two-hand tracking
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    # For swipe detection
    # Store history of index finger tip positions for each hand
    # hand_id -> deque of (x,y) tuples
    index_finger_history = [collections.deque(maxlen=10) for _ in range(2)]
    swipe_threshold_x = 0.07  # Min horizontal distance for swipe (normalized)
    swipe_threshold_y = 0.07  # Min vertical distance for swipe (normalized)
    swipe_debounce_frames = 5 # Number of frames to wait before detecting another swipe
    swipe_cooldown = [0, 0] # Cooldown counter for each hand

    # Pattern properties
    pattern_pos = (320, 240) # Initial X, Y position (center of a 640x480 frame)
    pattern_radius = 30
    pattern_color = (0, 0, 255) # Red in BGR
    available_colors = [
        (0, 0, 255),  # Red
        (0, 255, 0),  # Green
        (255, 0, 0),  # Blue
        (0, 255, 255), # Yellow
        (255, 0, 255), # Magenta
        (255, 255, 0)  # Cyan
    ]
    current_color_index = 0

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a later selfie-view display
        # Convert the BGR image to RGB
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        gesture = "None" # Variable to store detected gesture

        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                )

                # Gesture detection logic
                if len(hand_landmarks.landmark) > mp_hands.HandLandmark.INDEX_FINGER_TIP:
                    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                    # Pinch detection
                    pinch_distance = calculate_distance(thumb_tip, index_finger_tip)
                    pinch_threshold = 0.06 # This threshold might need tuning

                    # Initialize current_gesture_for_hand for this hand
                    current_gesture_for_hand = f"Hand {hand_idx}: Open"

                    if pinch_distance < pinch_threshold:
                        current_gesture_for_hand = f"Hand {hand_idx}: Pinch"
                        # Visual feedback for pinch
                        cv2.circle(image, (int(thumb_tip.x * image.shape[1]), int(thumb_tip.y * image.shape[0])),
                                   10, (0, 255, 0), -1)
                        cv2.circle(image, (int(index_finger_tip.x * image.shape[1]), int(index_finger_tip.y * image.shape[0])),
                                   10, (0, 255, 0), -1)

                        # Dynamic Size Control with Pinch:
                        # Map pinch_distance to radius. Max distance for pinch is pinch_threshold.
                        # Min distance is very small (e.g. 0.01).
                        # Scale factor might need adjustment. Let radius range from 5 to 100.
                        min_pinch_dist_for_radius = 0.01
                        max_pinch_dist_for_radius = pinch_threshold
                        # Ensure pinch_distance is within expected range for scaling
                        clamped_dist = max(min_pinch_dist_for_radius, min(pinch_distance, max_pinch_dist_for_radius))
                        # Scale normalized distance to pixel radius
                        # When distance is small (clamped_dist ~ min_pinch_dist), radius is small.
                        # When distance is large (clamped_dist ~ max_pinch_dist), radius is large.
                        new_radius = 5 + (clamped_dist - min_pinch_dist_for_radius) / \
                                     (max_pinch_dist_for_radius - min_pinch_dist_for_radius) * 95
                        pattern_radius = int(new_radius)


                    # Tap detection (only if not pinching)
                    elif len(hand_landmarks.landmark) > mp_hands.HandLandmark.MIDDLE_FINGER_TIP:
                        middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                        tap_distance = calculate_distance(index_finger_tip, middle_finger_tip)
                        tap_threshold = 0.05 # Threshold for tap, needs tuning

                        if tap_distance < tap_threshold:
                            current_gesture_for_hand = f"Hand {hand_idx}: Tap"
                            cv2.circle(image, (int(index_finger_tip.x * image.shape[1]), int(index_finger_tip.y * image.shape[0])),
                                       10, (255, 0, 0), -1)
                            cv2.circle(image, (int(middle_finger_tip.x * image.shape[1]), int(middle_finger_tip.y * image.shape[0])),
                                       10, (255, 0, 0), -1)
                            # Tap to move pattern to index finger tip
                            pattern_pos = (index_finger_tip.x * image.shape[1], index_finger_tip.y * image.shape[0])

                    gesture = current_gesture_for_hand # Update main gesture string

                    # Swipe Detection Logic (can override pinch/tap display for the frame it occurs)
                    index_finger_history[hand_idx].append((index_finger_tip.x, index_finger_tip.y))

                    if swipe_cooldown[hand_idx] > 0:
                        swipe_cooldown[hand_idx] -= 1

                    if len(index_finger_history[hand_idx]) == index_finger_history[hand_idx].maxlen and swipe_cooldown[hand_idx] == 0:
                        start_pos = index_finger_history[hand_idx][0]
                        end_pos = index_finger_history[hand_idx][-1]

                        dx = end_pos[0] - start_pos[0]
                        dy = end_pos[1] - start_pos[1]

                        swipe_detected_this_frame = False
                        if abs(dx) > swipe_threshold_x and abs(dx) > abs(dy) * 2: # Horizontal swipe
                            if dx > 0:
                                gesture = f"Hand {hand_idx}: Swipe Right"
                                current_color_index = (current_color_index + 1) % len(available_colors)
                            else:
                                gesture = f"Hand {hand_idx}: Swipe Left"
                                current_color_index = (current_color_index - 1 + len(available_colors)) % len(available_colors)
                            pattern_color = available_colors[current_color_index]
                            swipe_detected_this_frame = True
                        elif abs(dy) > swipe_threshold_y and abs(dy) > abs(dx) * 2: # Vertical swipe
                            # Vertical swipes currently don't change patterns, but gesture is detected
                            if dy > 0:
                                gesture = f"Hand {hand_idx}: Swipe Down"
                            else:
                                gesture = f"Hand {hand_idx}: Swipe Up"
                            # Example: Could use vertical swipe to change pattern type or animation speed later
                            swipe_detected_this_frame = True

                        if swipe_detected_this_frame:
                            swipe_cooldown[hand_idx] = swipe_debounce_frames
                            index_finger_history[hand_idx].clear()

                else: # if no landmarks for some reason for this hand_idx
                    if hand_idx < len(index_finger_history):
                        index_finger_history[hand_idx].clear()
                    # Ensure gesture is reset if hand disappears or landmarks lost
                    if gesture.startswith(f"Hand {hand_idx}"):
                        gesture = "None"


        # Display the detected gesture.
        # This will overwrite if two hands show gestures simultaneously.
        cv2.putText(image, f"Gesture: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # Draw the pattern
        # For now, pattern_pos is static. Later it can be dynamic.
        # Ensure pattern_pos components are integers for cv2.circle
        current_pattern_pos = (int(pattern_pos[0]), int(pattern_pos[1]))
        cv2.circle(image, current_pattern_pos, pattern_radius, pattern_color, -1) # -1 for filled circle

        # Decrement cooldowns even if no hands are detected this frame
        for i in range(len(swipe_cooldown)):
            if swipe_cooldown[i] > 0 and not results.multi_hand_landmarks:
                 swipe_cooldown[i] -=1


        cv2.imshow('GestureCanvas AI', image)

        key = cv2.waitKey(5) & 0xFF
        if key == 27:  # Press Esc to exit
            break
        elif key == ord('s'): # Press 's' to save image
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"gesture_canvas_{timestamp}.png"
            # Save the image that is being displayed (includes webcam, landmarks, pattern)
            cv2.imwrite(filename, image)
            print(f"Image saved as {filename}")
        elif key == ord('a') and llm_initialized: # Press 'a' to ask AI
            print("Requesting suggestion from LLM...")
            # Example prompt. This could be made more dynamic later.
            prompt = "Suggest a calming color palette (3 colors in BGR format like (B,G,R), (B,G,R), (B,G,R)) and a simple geometric pattern type (like 'spiral', 'mandala', 'waves') for a desktop wallpaper."
            suggestion = get_llm_suggestion(prompt)
            print(f"\nLLM Suggestion:\n{suggestion}\n")
            # Optional: Add code here to parse suggestion and apply it to the pattern
            # For now, just printing. We could also display it on the OpenCV window.
            # To display on screen, you might need to handle multi-line text.
            # Example: cv2.putText(image, "AI: Check console", (10, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)


    hands.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
