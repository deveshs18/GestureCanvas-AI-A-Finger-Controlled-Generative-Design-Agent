import cv2
import mediapipe as mp
import numpy as np
import math # Keep math for potential distance calculations
import time # For timestamped filenames
import os
import base64
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Global variables for drawing
drawing_color = (255, 255, 255)  # Default: White (BGR)
canvas = None
prev_point = None # Previous point for drawing lines
selected_color_index = 0 # To keep track of which color in the palette is active

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

# --- OpenAI Vision Integration ---
llm = None

def init_llm():
    global llm
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not found. AI features will be disabled.")
        return False
    try:
        # Use a vision-capable model
        llm = ChatOpenAI(model="gpt-4o", max_tokens=100)
        print("OpenAI Vision model (gpt-4o) initialized successfully.")
        return True
    except Exception as e:
        print(f"Error initializing OpenAI Vision model: {e}")
        llm = None
        return False

def get_openai_suggestion(image):
    if not llm:
        return "LLM not initialized."

    # Encode the image to base64
    _, buffer = cv2.imencode(".jpg", image)
    image_base64 = base64.b64encode(buffer).decode("utf-8")

    try:
        msg = llm.invoke(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": "Analyze the user's facial expression in this image. Based on their dominant emotion, suggest a palette of 3 colors. List the colors as BGR tuples, for example: (B,G,R), (B,G,R), (B,G,R)."},
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{image_base64}",
                        },
                    ]
                )
            ]
        )
        return msg.content
    except Exception as e:
        return f"Error getting suggestion from OpenAI: {e}"
# --- End OpenAI Integration ---

def main():
    global canvas, drawing_color, prev_point, palette_rects, selected_color_index

    llm_initialized = init_llm()
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
        for i, (x, y, w, h) in enumerate(palette_rects):
            cv2.rectangle(display_image, (x, y), (x + w, y + h), palette_colors[i], -1) # Filled rectangle

            # Add highlight to the selected color
            if i == selected_color_index:
                # Cyan border for selection
                cv2.rectangle(display_image, (x - 2, y - 2), (x + w + 2, y + h + 2), (255, 255, 0), 2)

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
