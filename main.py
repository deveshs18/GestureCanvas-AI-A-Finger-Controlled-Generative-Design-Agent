import cv2
import mediapipe as mp
import numpy as np
import math

# --- Global State for Tessellation ---
# Shape properties
available_shapes = ['hexagon', 'triangle', 'square']
shape_index = 0
current_shape = available_shapes[shape_index]
shape_size = 50.0
base_color = (255, 0, 0) # Default Blue (BGR)
# For pinch-to-change-shape gesture
right_hand_pinching = False
pinch_cooldown = 0

# Transformation properties
grid_offset_x = 0
grid_offset_y = 0
zoom = 1.0
# --- End Global State ---

def get_shape_vertices(shape_type, center, size):
    """Calculates the vertices for a regular polygon."""
    x, y = center
    vertices = []
    if shape_type == 'hexagon':
        num_sides = 6
        # Start angle needs to be 30 degrees to make hexagons flat top/bottom
        start_angle = 30
        for i in range(num_sides):
            angle = math.radians(60 * i + start_angle)
            vx = x + size * math.cos(angle)
            vy = y + size * math.sin(angle)
            vertices.append((vx, vy))
    elif shape_type == 'triangle':
        num_sides = 3
        for i in range(num_sides):
            angle = math.radians(120 * i - 90) # Pointing up
            vx = x + size * math.cos(angle)
            vy = y + size * math.sin(angle)
            vertices.append((vx, vy))
    elif shape_type == 'square':
        # More direct calculation for a square
        half_size = size / math.sqrt(2) # To keep apparent size consistent
        vertices.append((x - half_size, y - half_size))
        vertices.append((x + half_size, y - half_size))
        vertices.append((x + half_size, y + half_size))
        vertices.append((x - half_size, y + half_size))

    return np.array(vertices, dtype=np.int32)

def draw_tessellation(canvas, shape_type, size, color, offset_x, offset_y):
    """Draws a tessellating grid of shapes on the canvas."""
    h, w, _ = canvas.shape

    if shape_type == 'hexagon':
        # Hexagonal grid layout calculations
        hex_width = math.sqrt(3) * size
        hex_height = 2 * size

        # Calculate how many hexagons to draw in each direction
        cols = int(w / hex_width) + 2
        rows = int(h / (hex_height * 0.75)) + 2

        for row in range(rows):
            for col in range(cols):
                # Calculate center of each hexagon
                x_center = col * hex_width + (row % 2) * (hex_width / 2)
                y_center = row * hex_height * 0.75

                # Apply global offset for movement illusion
                x_center = (x_center + offset_x) % (w + hex_width) - hex_width/2
                y_center = (y_center + offset_y) % (h + hex_height*0.75) - hex_height/2

                # Get vertices and draw the hexagon
                vertices = get_shape_vertices('hexagon', (x_center, y_center), size)
                cv2.fillPoly(canvas, [vertices], color)
    elif shape_type == 'square':
        square_size = int(size * 1.414) # Adjust size for visual consistency
        cols = int(w / square_size) + 2
        rows = int(h / square_size) + 2
        for row in range(rows):
            for col in range(cols):
                x_center = col * square_size
                y_center = row * square_size
                x_center = (x_center + offset_x) % (w + square_size) - square_size
                y_center = (y_center + offset_y) % (h + square_size) - square_size
                vertices = get_shape_vertices('square', (x_center, y_center), size)
                cv2.fillPoly(canvas, [vertices], color)
    elif shape_type == 'triangle':
        tri_height = size * 2
        tri_width = size * math.sqrt(3)
        cols = int(w / tri_width) + 2
        rows = int(h / tri_height) + 2
        for row in range(rows):
            for col in range(cols):
                x_center = col * tri_width + (row % 2) * (tri_width / 2)
                y_center = row * tri_height * 0.75 # A bit of overlap
                x_center = (x_center + offset_x) % (w + tri_width) - tri_width
                y_center = (y_center + offset_y) % (h + tri_height) - tri_height
                vertices = get_shape_vertices('triangle', (x_center, y_center), size)
                cv2.fillPoly(canvas, [vertices], color)
                # Inverted triangle for perfect tiling
                vertices_inv = get_shape_vertices('triangle', (x_center + tri_width/2, y_center), size)
                vertices_inv = cv2.transform(np.array([vertices_inv]), cv2.getRotationMatrix2D((x_center, y_center), 180, 1.0))[0]
                cv2.fillPoly(canvas, [vertices_inv], color)


def main():
    global shape_index, current_shape, shape_size, base_color, grid_offset_x, grid_offset_y, right_hand_pinching, pinch_cooldown
    print("Motion Tessellation Engine Starting...")

    # Initialize MediaPipe Hands for two-hand tracking
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip for a selfie-view and get frame dimensions
        image = cv2.flip(image, 1)
        h, w, _ = image.shape

        # Create a black canvas for drawing the tessellation
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

        # --- Tessellation Drawing Logic ---
        draw_tessellation(canvas, current_shape, shape_size, base_color, grid_offset_x, grid_offset_y)

        # The final image will be a combination of the canvas and the hand landmarks.
        display_image = canvas

        # --- Hand Tracking Logic ---
        # Convert the BGR image to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_image)

        if results.multi_hand_landmarks:
            # Differentiate between left and right hands
            hand_data = {'left': None, 'right': None}
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_type = handedness.classification[0].label.lower()
                hand_data[hand_type] = hand_landmarks

            # --- Right Hand Controls (Shape, Color, Size) ---
            if hand_data['right']:
                landmarks = hand_data['right'].landmark
                # Position for Color
                # Map X to Hue (0-179), Y to Value (0-255)
                h_val = int(landmarks[0].x * 179)
                v_val = int(landmarks[0].y * 255)
                hsv_color = np.uint8([[[h_val, 255, v_val]]])
                bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
                base_color = tuple(map(int, bgr_color)) # Convert numpy types to int

                # Pinch for Size and Shape Change
                thumb_tip = landmarks[mp.solutions.hands.HandLandmark.THUMB_TIP]
                index_tip = landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
                pinch_dist = math.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)

                # Pinch Distance for Size (map 0.01 to 0.2 -> 10px to 100px)
                min_pinch, max_pinch = 0.01, 0.2
                min_size, max_size = 10.0, 100.0
                clamped_dist = max(min_pinch, min(pinch_dist, max_pinch))
                shape_size = min_size + ((clamped_dist - min_pinch) / (max_pinch - min_pinch)) * (max_size - min_size)

                # Pinch Click for Shape Change
                pinch_threshold = 0.05
                if pinch_dist < pinch_threshold and not right_hand_pinching and pinch_cooldown == 0:
                    # Just pinched
                    right_hand_pinching = True
                    shape_index = (shape_index + 1) % len(available_shapes)
                    current_shape = available_shapes[shape_index]
                    pinch_cooldown = 15 # Wait 15 frames before another pinch
                elif pinch_dist >= pinch_threshold:
                    right_hand_pinching = False

            if pinch_cooldown > 0:
                pinch_cooldown -= 1

            # --- Left Hand Controls (Movement) ---
            if hand_data['left']:
                landmarks = hand_data['left'].landmark
                # Use wrist position for stable control
                wrist_pos = landmarks[mp.solutions.hands.HandLandmark.WRIST]

                # Center of the screen
                center_screen_x = w / 2
                center_screen_y = h / 2

                # Normalized hand position relative to center
                # Values will be roughly -0.5 to 0.5
                control_x = wrist_pos.x - (center_screen_x / w)
                control_y = wrist_pos.y - (center_screen_y / h)

                # Define a speed factor for the movement
                speed_factor = 50

                # Update grid offset based on hand position relative to center
                # The further from the center, the faster the movement
                grid_offset_x += control_x * speed_factor
                grid_offset_y += control_y * speed_factor


            # Draw landmarks for feedback
            for hand_type, hand_landmarks in hand_data.items():
                if hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # To see hands for now, let's overlay the webcam feed with landmarks on the canvas
        # This is for debugging/development. Final version might just show the tessellation.
        display_image = cv2.addWeighted(display_image, 1, image, 0.5, 0)

        # --- Add UI Text ---
        shape_text = f"Shape: {current_shape}"
        cv2.putText(display_image, shape_text, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


        cv2.imshow('Motion Tessellations', display_image)

        if cv2.waitKey(5) & 0xFF == 27:  # Press Esc to exit
            break

    hands.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
