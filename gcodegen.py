#! /usr/bin/python3

import cv2,sys
import numpy as np
from PIL import Image, ImageOps

# --- CONFIGURATION ---
INPUT_IMAGE_FILE = sys.argv[1]

try:
    SCALE_FACTOR=sys.argv[2]
    SCALE_FACTOR=eval(SCALE_FACTOR)
except: SCALE_FACTOR=1

try: LASER_POWER='S'+sys.argv[3]    
except: LASER_POWER='S975'
try: FEED_RATE='F'+sys.argv[4]    
except: FEED_RATE='F2400'
try: FSPEED='F'+sys.argv[5]    
except: FSPEED='F2400'

#LASER_POWER = 'S255'                 # Laser power command (e.g., S255 for max)
#FEED_RATE = 'F1200'                  # Movement speed in mm/min

GCODE_OUTPUT_FILE = 'gcode.gcode'

def generate_gcode(contours, scale_factor=SCALE_FACTOR):
    """
    Generates G-code from a list of OpenCV contour matrices.
    
    Args:
        contours (list): A list of NumPy arrays, where each array is a contour.
        scale_factor (float): Multiplier to convert pixel coordinates to mm.
                              Adjust this based on your image DPI/desired scale.
                              (e.g., 1 pixel = 0.1mm)

    Returns:
        str: The complete G-code program.
    """
    gcode = []
    
    # G-code initialization (common for laser/CNC)
    gcode.append('G21 ; Set units to millimeters')
    gcode.append('G90 ; Use absolute positioning')
    gcode.append('G17 ; Select XY plane')
    gcode.append('M5  ; Turn off laser to start')
#    gcode.append(f'G00 Z1.00 ; Lift laser for safe travel (1mm)')
    
    # Process each contour (each trace/boundary)
    for i, contour in enumerate(contours):
        # Flatten the NumPy array to N x 2 matrix (removes the extra dim)
        # Shape goes from (N, 1, 2) to (N, 2)
        points = contour.squeeze() 

        # Ensure we have at least one point
        if points.ndim == 1:
            points = np.expand_dims(points, axis=0)
        if points.size == 0:
            continue

        # --- Start of a new tool path ---
        # The first move is a fast G0 move with the laser OFF
        first_x = points[0, 0] * scale_factor
        first_y = points[0, 1] * scale_factor
        gcode.append(f'\n; Contour {i+1}')
        gcode.append(f'G00 X{first_x:.3f} Y{first_y:.3f} {FSPEED}; Rapid move to start point')

        # --- Begin burning/cutting ---
        # Move laser down to focus height and start burning
#        gcode.append(f'G01 Z0.00 ; Lower Z to cutting height') noko
        gcode.append(f'M3 {LASER_POWER} ; Turn on laser with power')
        
        # All subsequent moves follow the contour line
        # G01 is a controlled linear move command
        for x_px, y_px in points:
            x_mm = x_px * scale_factor
            y_mm = y_px * scale_factor
            gcode.append(f'G01 X{x_mm:.3f} Y{y_mm:.3f} {FEED_RATE}')

        # --- End of tool path ---
        # Lift laser and stop burning/cutting
        gcode.append(f'M5 ; Turn off laser')
        gcode.append(f'G00 Z1.00 ; Lift Z for clearance')

    # Final commands
    gcode.append('\nG00 X0 Y0 ; Return to origin')
    gcode.append('M30 ; Program end')
    
    return '\n'.join(gcode)

# ----------------- MAIN EXECUTION -----------------

# 1. Image Preprocessing (The core of finding contours)
try:
    # Use PIL/Pillow for robust image opening and to get image dimensions
    img = Image.open(INPUT_IMAGE_FILE).convert('RGB')
    img = ImageOps.flip(img)
    width, height = img.size
    
    # Convert PIL image to OpenCV format (NumPy array)
    image_np = np.array(img)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    
    # Apply a Gaussian blur to reduce noise and improve contour detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use adaptive thresholding to better handle variations in the image
    binary = cv2.adaptiveThreshold(
        blurred, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        11, 2
    )
    
    # 2. Find Contours
    # RETR_LIST: Retrieves all contours without any hierarchy.
    # CHAIN_APPROX_NONE: Stores all contour points.
    contours, _ = cv2.findContours(
        binary, 
        cv2.RETR_LIST, 
        cv2.CHAIN_APPROX_NONE
    )

    print(f"Found {len(contours)} contours in the image.")
    # 

    # 3. Generate and Save G-code
    final_gcode = generate_gcode(contours)
    
    with open(GCODE_OUTPUT_FILE, 'w') as f:
        f.write(final_gcode)
        
    print(f"Successfully created G-code file: {GCODE_OUTPUT_FILE}")
    
except FileNotFoundError:
    print(f"Error: Input file '{INPUT_IMAGE_FILE}' not found. Please check the file name.")
except Exception as e:
    print(f"An error occurred: {e}")

