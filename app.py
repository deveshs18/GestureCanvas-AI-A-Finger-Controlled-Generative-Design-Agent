import streamlit as st
import cv2
import numpy as np
from main import (
    available_shapes,
    get_shape_vertices,
    draw_tessellation,
    bgr_to_hex,
    init_llm,
    apply_ai_theme,
    reset_scene,
    DEFAULT_SHAPE_INDEX,
    DEFAULT_SHAPE_SIZE,
    DEFAULT_BASE_COLOR,
    DEFAULT_SPEED_FACTOR,
)

def main():
    st.title("Gesture-Controlled Generative Art")

    # Initialize session state
    if "shape_index" not in st.session_state:
        st.session_state.shape_index = DEFAULT_SHAPE_INDEX
    if "shape_size" not in st.session_state:
        st.session_state.shape_size = DEFAULT_SHAPE_SIZE
    if "base_color" not in st.session_state:
        st.session_state.base_color = DEFAULT_BASE_COLOR
    if "speed_factor" not in st.session_state:
        st.session_state.speed_factor = DEFAULT_SPEED_FACTOR
    if "grid_offset_x" not in st.session_state:
        st.session_state.grid_offset_x = 0
    if "grid_offset_y" not in st.session_state:
        st.session_state.grid_offset_y = 0
    if "ai_explanation" not in st.session_state:
        st.session_state.ai_explanation = "Ask the AI for a theme!"

    # Sidebar controls
    st.sidebar.header("Controls")
    st.session_state.shape_index = st.sidebar.selectbox(
        "Shape",
        range(len(available_shapes)),
        format_func=lambda x: available_shapes[x].capitalize(),
        index=st.session_state.shape_index,
    )
    current_shape = available_shapes[st.session_state.shape_index]

    st.session_state.shape_size = st.sidebar.slider(
        "Size", 10.0, 100.0, st.session_state.shape_size
    )
    hex_color = st.sidebar.color_picker(
        "Color", bgr_to_hex(st.session_state.base_color)
    )
    st.session_state.base_color = tuple(int(hex_color.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))

    st.session_state.speed_factor = st.sidebar.slider(
        "Speed", 10.0, 100.0, st.session_state.speed_factor
    )

    if st.sidebar.button("Reset Scene"):
        reset_scene()
        st.session_state.ai_explanation = "Ask the AI for a theme!"
        st.experimental_rerun()

    if st.sidebar.button("Generate AI Theme"):
        init_llm()
        apply_ai_theme()
        st.experimental_rerun()

    # Main canvas
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    draw_tessellation(
        canvas,
        current_shape,
        st.session_state.shape_size,
        st.session_state.base_color,
        st.session_state.grid_offset_x,
        st.session_state.grid_offset_y,
    )

    st.image(canvas, channels="BGR", use_column_width=True)

    st.info(st.session_state.ai_explanation)

if __name__ == "__main__":
    main()
