"""
Gesture-Controlled Generative Art System with OpenAI Assistant

This script launches a real-time interactive art application where users can control
a generative tessellation pattern using hand gestures via a webcam.

Features:
- Two-Handed Gesture Control (MediaPipe):
  - Right Hand: Controls shape type, color, and size.
  - Left Hand: Controls pattern movement/offset.
- Dynamic Tessellation Engine (OpenCV):
  - Renders a seamless grid of hexagons, triangles, or squares.
- OpenAI-Powered Theme Generator (LangChain):
  - Press 'A' to ask an AI for a new theme (shape, color, size, speed).
- Save and Reset:
  - Press 'S' to save the artwork to the 'screenshots/' folder.
  - Press 'R' to reset the scene to its default state.
  - Press 'ESC' to exit.
"""
import cv2
import mediapipe as mp
import numpy as np
import math
import os
import time
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import re

# --- Global State and Defaults ---
# Default values for resetting the scene
DEFAULT_SHAPE_INDEX = 0
DEFAULT_SHAPE_SIZE = 50.0
DEFAULT_BASE_COLOR = (255, 0, 0)  # Default Blue (BGR)
DEFAULT_SPEED_FACTOR = 50.0

# Live state variables
available_shapes = ['hexagon', 'triangle', 'square']
shape_index = DEFAULT_SHAPE_INDEX
current_shape = available_shapes[shape_index]
shape_size = DEFAULT_SHAPE_SIZE
base_color = DEFAULT_BASE_COLOR
speed_factor = DEFAULT_SPEED_FACTOR

# Gesture control state
right_hand_pinching = False
pinch_cooldown = 0

# Transformation properties
grid_offset_x = 0
grid_offset_y = 0
zoom = 1.0
# --- End Global State ---

# --- OpenAI Integration ---
llm = None

def init_llm():
    """Initializes the OpenAI LLM from environment variables."""
    global llm
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY not found in .env file. AI features will be disabled.")
        return False
    try:
        llm = ChatOpenAI(model="gpt-3.5-turbo", max_tokens=150)
        print("OpenAI LLM initialized successfully.")
        return True
    except Exception as e:
        print(f"Error initializing OpenAI LLM: {e}")
        return False

def parse_ai_response(response):
    """Parses the LLM's text response to extract theme parameters using regex."""
    theme = {}
    try:
        # Extract shape
        shape_match = re.search(r"shape:\s*(\w+)", response, re.IGNORECASE)
        if shape_match and shape_match.group(1).lower() in available_shapes:
            theme['shape'] = shape_match.group(1).lower()

        # Extract base color (first BGR tuple found)
        color_match = re.search(r"\((\d{1,3}),\s*(\d{1,3}),\s*(\d{1,3})\)", response)
        if color_match:
            b, g, r = int(color_match.group(1)), int(color_match.group(2)), int(color_match.group(3))
            theme['color'] = (b, g, r)

        # Extract size
        size_match = re.search(r"size:\s*(\w+)", response, re.IGNORECASE)
        if size_match:
            size_map = {'small': 30.0, 'medium': 60.0, 'large': 90.0}
            theme['size'] = size_map.get(size_match.group(1).lower(), DEFAULT_SHAPE_SIZE)

        # Extract motion speed
        speed_match = re.search(r"speed:\s*(\w+)", response, re.IGNORECASE)
        if speed_match:
            speed_map = {'slow': 20.0, 'medium': 50.0, 'fast': 80.0}
            theme['speed'] = speed_map.get(speed_match.group(1).lower(), DEFAULT_SPEED_FACTOR)

        # Extract explanation
        explanation_match = re.search(r"explanation:\s*(.*)", response, re.IGNORECASE)
        if explanation_match:
            theme['explanation'] = explanation_match.group(1).strip()

    except Exception as e:
        print(f"Error parsing AI response: {e}")
    return theme

def apply_ai_theme():
    """Gets a theme from the LLM, parses it, and applies it to the scene."""
    global current_shape, shape_index, base_color, shape_size, speed_factor, ai_explanation
    if not llm:
        print("AI not initialized. Cannot apply theme.")
        return

    print("Requesting AI theme from OpenAI...")
    prompt = """
    Generate a generative art theme. Your response must be formatted exactly as follows, with each item on a new line:
    - shape: [one of hexagon, triangle, or square]
    - color: [a single BGR tuple, e.g., (120, 50, 200)]
    - size: [one of small, medium, or large]
    - speed: [one of slow, medium, or fast]
    - explanation: [a brief, creative explanation for the theme choice]
    """
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        print(f"AI Response:\n{response.content}")
        theme = parse_ai_response(response.content)

        if not theme:
            print("Could not parse a valid theme from the AI response.")
            return

        print(f"Applying theme: {theme}")
        if 'shape' in theme:
            current_shape = theme['shape']
            shape_index = available_shapes.index(current_shape)
        if 'color' in theme:
            base_color = theme['color']
        if 'size' in theme:
            shape_size = theme['size']
        if 'speed' in theme:
            speed_factor = theme['speed']
        if 'explanation' in theme:
            ai_explanation = theme['explanation']
    except Exception as e:
        print(f"Error communicating with OpenAI: {e}")
# --- End OpenAI Integration ---

# --- Helper Functions ---
def reset_scene():
    """Resets the scene to its default state."""
    global shape_index, current_shape, shape_size, base_color, speed_factor, grid_offset_x, grid_offset_y
    print("Resetting scene to defaults.")
    shape_index = DEFAULT_SHAPE_INDEX
    current_shape = available_shapes[shape_index]
    shape_size = DEFAULT_SHAPE_SIZE
    base_color = DEFAULT_BASE_COLOR
    speed_factor = DEFAULT_SPEED_FACTOR
    grid_offset_x = 0
    grid_offset_y = 0

def bgr_to_hex(bgr_color):
    """Converts a BGR color tuple to a hex string."""
    return f"#{bgr_color[2]:02X}{bgr_color[1]:02X}{bgr_color[0]:02X}"

def get_shape_vertices(shape_type, center, size):
    """Calculates the vertices for a regular polygon."""
    x, y = center
    vertices = []
    if shape_type == 'hexagon':
        num_sides = 6
        start_angle = 30  # Flat top
        for i in range(num_sides):
            angle = math.radians(60 * i + start_angle)
            vx = x + size * math.cos(angle)
            vy = y + size * math.sin(angle)
            vertices.append((vx, vy))
    elif shape_type == 'triangle':
        num_sides = 3
        for i in range(num_sides):
            angle = math.radians(120 * i - 90)  # Pointing up
            vx = x + size * math.cos(angle)
            vy = y + size * math.sin(angle)
            vertices.append((vx, vy))
    elif shape_type == 'square':
        half_size = size / math.sqrt(2)
        vertices.extend([
            (x - half_size, y - half_size), (x + half_size, y - half_size),
            (x + half_size, y + half_size), (x - half_size, y + half_size)
        ])
    return np.array(vertices, dtype=np.int32)
# --- End Helper Functions ---

# --- Drawing Engine ---
def draw_tessellation(canvas, shape_type, size, color, offset_x, offset_y):
    """Draws a tessellating grid of shapes on the canvas."""
    h, w, _ = canvas.shape
    if size < 1: return # Avoid division by zero or excessive computation

    if shape_type == 'hexagon':
        hex_width = math.sqrt(3) * size
        hex_height = 2 * size
        cols = int(w / hex_width) + 2
        rows = int(h / (hex_height * 0.75)) + 2
        for row in range(rows):
            for col in range(cols):
                x_center = col * hex_width + (row % 2) * (hex_width / 2)
                y_center = row * hex_height * 0.75
                x_center_offset = (x_center + offset_x) % (w + hex_width) - hex_width
                y_center_offset = (y_center + offset_y) % (h + hex_height) - hex_height
                vertices = get_shape_vertices('hexagon', (x_center_offset, y_center_offset), size)
                cv2.fillPoly(canvas, [vertices], color)
    elif shape_type == 'square':
        square_size = int(size * 1.414)
        cols = int(w / square_size) + 2
        rows = int(h / square_size) + 2
        for row in range(rows):
            for col in range(cols):
                x_center = col * square_size
                y_center = row * square_size
                x_center_offset = (x_center + offset_x) % (w + square_size) - square_size
                y_center_offset = (y_center + offset_y) % (h + square_size) - square_size
                vertices = get_shape_vertices('square', (x_center_offset, y_center_offset), size)
                cv2.fillPoly(canvas, [vertices], color)
    elif shape_type == 'triangle':
        tri_height = size * 1.5
        tri_width = size * math.sqrt(3)
        cols = int(w / tri_width) + 2
        rows = int(h / tri_height) + 2
        for row in range(rows):
            for col in range(cols):
                x_center = col * tri_width + (row % 2) * (tri_width / 2)
                y_center = row * tri_height
                x_center_offset = (x_center + offset_x) % (w + tri_width) - tri_width
                y_center_offset = (y_center + offset_y) % (h + tri_height) - tri_height
                vertices = get_shape_vertices('triangle', (x_center_offset, y_center_offset), size)
                cv2.fillPoly(canvas, [vertices], color)
                # Inverted triangle for perfect tiling
                vertices_inv = get_shape_vertices('triangle', (x_center_offset + tri_width / 2, y_center_offset + tri_height / 2), size)
                vertices_inv = cv2.transform(np.array([vertices_inv]), cv2.getRotationMatrix2D((x_center_offset, y_center_offset), 180, 1.0))[0]
                cv2.fillPoly(canvas, [vertices_inv], color)
# --- End Drawing Engine ---
