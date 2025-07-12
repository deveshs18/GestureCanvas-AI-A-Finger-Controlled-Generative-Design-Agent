import cv2
import mediapipe as mp
import numpy as np
import math # Keep math for potential distance calculations
from deepface import DeepFace # For facial emotion detection
import time # For timestamped filenames

# Global variables for drawing
drawing_color = (255, 255, 255)  # Default: White (BGR)
canvas = None
prev_point = None # Previous point for drawing lines
selected_color_index = 0 # To keep track of which color in the palette is active

# --- Emotion Detection Globals ---
emotion_analysis_interval = 90  # Analyze emotion every 90 frames (approx. 3 seconds at 30fps)
frame_counter = 0
detected_emotion = "neutral"
emotion_color_suggestions = {
    "happy": [4, 2],  # Suggest Yellow and Green
    "sad": [3],       # Suggest Blue
    "angry": [1],     # Suggest Red
    "surprise": [4],  # Suggest Yellow
    "neutral": [0],   # Suggest White
    "fear": [3],      # Suggest Blue
    "disgust": [2],   # Suggest Green
}
# --- End Emotion Detection Globals ---

# Define the color palette (BGR format)
# (More colors can be added)
palette_colors = [
    (255, 255, 255),  # White
    (0, 0, 255),      # Red
    (0, 255, 0),      # Green
    (255, 0, 0),      # Blue
    (0, 255, 255),    # Yellow
]
palette_rects = [] # Will store (x, y, w, h) for each color rectangle
palette_rect_size = 30 # Size of each color square in the palette
palette_margin = 10 # Margin around palette and between squares

def main():
    global canvas, drawing_color, prev_point, palette_rects, selected_color_index
    global frame_counter, detected_emotion

    print("Gesture Drawing Canvas Starting...")

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    # Focusing on one hand for drawing simplicity for now
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Initialize canvas once we know frame dimensions
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        cap.release()
        return

    # Create a blank canvas for drawing (3 channels for BGR color)
    # Initialize with black - user will draw with 'drawing_color' (e.g. white)
    canvas = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)

    # --- Initialize Palette Rectangles ---
    # Position the palette at the top of the screen
    for i, color in enumerate(palette_colors):
        x = palette_margin + i * (palette_rect_size + palette_margin)
        y = palette_margin
        palette_rects.append((x, y, palette_rect_size, palette_rect_size))
    # --- End Palette Init ---

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a later selfie-view display
        image = cv2.flip(image, 1)

        # --- Emotion Detection Logic ---
        frame_counter += 1
        if frame_counter >= emotion_analysis_interval:
            frame_counter = 0
            try:
                # DeepFace expects BGR, and 'image' is already in BGR format
                analysis = DeepFace.analyze(image, actions=['emotion'], enforce_detection=False)
                # DeepFace returns a list of dicts, one for each face. We'll use the first one.
                if isinstance(analysis, list) and len(analysis) > 0:
                    detected_emotion = analysis[0]['dominant_emotion']
                    print(f"Detected Emotion: {detected_emotion}") # For debugging
            except Exception as e:
                # This can happen if a model file is missing or other issues arise
                print(f"Error during emotion analysis: {e}")
        # --- End Emotion Detection Logic ---

        # Convert the BGR image to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image and find hands
        results = hands.process(rgb_image)

        # Convert the RGB image back to BGR for OpenCV display
        # image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR) # This line is not needed if we use the original 'image'

        # Hand landmark drawing and gesture logic will go here in future steps

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the original image (for user feedback)
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4), # Landmark color
                    mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2) # Connection color
                )

                # --- Drawing Logic ---
                # Get coordinates of index finger tip and thumb tip
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

                # Convert normalized coordinates to pixel coordinates
                h, w, _ = image.shape
                center_x, center_y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

                # Calculate distance between thumb and index finger
                pinch_distance = math.sqrt(
                    (index_finger_tip.x - thumb_tip.x)**2 +
                    (index_finger_tip.y - thumb_tip.y)**2
                )

                # Define drawing gesture: pinch gesture
                pinch_threshold = 0.08 # Threshold to start drawing

                # Check for color selection first
                color_selected = False
                if pinch_distance < pinch_threshold:
                    for i, (x, y, w, h) in enumerate(palette_rects):
                        if x < center_x < x + w and y < center_y < y + h:
                            drawing_color = palette_colors[i]
                            selected_color_index = i
                            color_selected = True
                            # When selecting a color, don't draw a point
                            prev_point = None
                            break # Exit after finding the selected color

                if pinch_distance < pinch_threshold and not color_selected:
                    # Drawing mode is active
                    if prev_point is None:
                        # Start of a new line
                        prev_point = (center_x, center_y)

                    # Draw a line from the previous point to the current point
                    cv2.line(canvas, prev_point, (center_x, center_y), drawing_color, thickness=5)
                    prev_point = (center_x, center_y)
                else:
                    # Not in drawing mode, reset previous point
                    prev_point = None
        else:
            # If no hands are detected, reset previous point
            prev_point = None


        # --- Overlay drawing canvas on the main image ---
        if canvas is not None:
            img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            _, drawn_mask = cv2.threshold(img_gray, 10, 255, cv2.THRESH_BINARY)
            drawn_mask_inv = cv2.bitwise_not(drawn_mask)
            img_bg = cv2.bitwise_and(image, image, mask=drawn_mask_inv)
            img_fg = cv2.bitwise_and(canvas, canvas, mask=drawn_mask)
            display_image = cv2.add(img_bg, img_fg)
        else:
            display_image = image

        # --- Draw the Palette UI on top of everything ---
        suggested_indices = emotion_color_suggestions.get(detected_emotion, [])
        for i, (x, y, w, h) in enumerate(palette_rects):
            cv2.rectangle(display_image, (x, y), (x + w, y + h), palette_colors[i], -1) # Filled rectangle

            # Add highlight for recommended colors
            if i in suggested_indices:
                # Green border for suggestion
                cv2.rectangle(display_image, (x - 4, y - 4), (x + w + 4, y + h + 4), (0, 255, 0), 2)

            # Add highlight to the selected color (drawn on top of suggestion highlight if necessary)
            if i == selected_color_index:
                # Cyan border for selection
                cv2.rectangle(display_image, (x - 2, y - 2), (x + w + 2, y + h + 2), (255, 255, 0), 2)

        # --- Display Detected Emotion ---
        emotion_text = f"Emotion: {detected_emotion}"
        text_pos = (palette_margin, display_image.shape[0] - palette_margin - 10) # Bottom-left
        cv2.putText(display_image, emotion_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)


        cv2.imshow('Gesture Drawing Canvas', display_image)

        key = cv2.waitKey(5) & 0xFF
        if key == 27:  # Press Esc to exit
            break
        elif key == ord('s'): # Press 's' to save the drawing
            if canvas is not None:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = f"drawing_{timestamp}.png"
                # Save the drawing canvas only, not the webcam feed or UI
                cv2.imwrite(filename, canvas)
                print(f"Drawing saved as {filename}")

    hands.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
