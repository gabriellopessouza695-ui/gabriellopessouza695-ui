import os
import cv2
import numpy as np
import ot  # Python Optimal Transport (POT)
from xml.etree import ElementTree as ET

# --- CONFIGURATION ---
PALETTE = {
    "dark": "#A78BFA",
    "light": "#7C3AED",
    "chrome_dark": "#22D3EE",
    "chrome_light": "#0891B2",
    "accent": "#10B981",
    "bg": "#0A101F"
}

USER_DATA = {
    "Subject": "Gabriel Souza Lopes",
    "Role": "Backend Developer",
    "Origin": "João Monlevade, Brazil",
    "Education": "Sistemas de Informação @ UFOP",
    "Status": "Building + Learning + Shipping",
    "ToolChain": "VS Code, Git, N8n, ClickUp",
    "Core.Lang": "Java, C, JavaScript",
    "Core.Frontend": "HTML, CSS",
    "Core.Backend": "Spring Boot, RabbitMQ",
    "Core.Database": "PostgreSQL, MySQL",
    "Core.Infra": "Docker, GitHub Actions",
    "Grid.Mail": "gabriellopessouza695@gmail.com",
    "Grid.Portfolio": "Coming Soon",
    "Grid.LinkedIn": "gabriel-lopes-90422a241",
    "Grid.GitHub": "gabriellopessouza695",
    "Grid.Instagram": "souza_dev06"
}

SVG_WIDTH = 1180
SVG_HEIGHT = 610
PORTRAIT_WIDTH = int(SVG_WIDTH * 0.38)

def compute_dotted_leaders(label, value, total_width=600, char_width=8):
    """Calculates the exact number of dots needed to fill the space between label and value."""
    text_len = len(label) + len(value)
    available_space = (total_width - (text_len * char_width))
    dots_needed = max(3, available_space // char_width)
    return "." * dots_needed

def process_portrait(image_path, mode="dark"):
    """
    Applies autocontrast, UnsharpMask, and 1-bit Floyd-Steinberg dithering.
    Dark mode applies binary closing to segment the background.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load {image_path}")
    
    img = cv2.resize(img, (300, 340))
    
    # Contrast 1.3x and Autocontrast
    img = cv2.convertScaleAbs(img, alpha=1.3, beta=0)
    
    # Unsharp Mask (radius=3, percent=140)
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.addWeighted(img, 2.4, blurred, -1.4, 0)
    
    if mode == "dark":
        # Background segmentation: Threshold, binary closing, keep largest component
        _, thresh = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        img = cv2.bitwise_and(img, img, mask=mask)
    
    # Floyd-Steinberg Dithering
    h, w = img.shape
    dithered = img.copy().astype(float)
    
    for y in range(h):
        for x in range(w):
            old_pixel = dithered[y, x]
            new_pixel = 255 if old_pixel > 127 else 0
            dithered[y, x] = new_pixel
            quant_error = old_pixel - new_pixel
            
            if x + 1 < w:
                dithered[y, x + 1] += quant_error * 7 / 16
            if y + 1 < h:
                if x > 0:
                    dithered[y + 1, x - 1] += quant_error * 3 / 16
                dithered[y + 1, x] += quant_error * 5 / 16
                if x + 1 < w:
                    dithered[y + 1, x + 1] += quant_error * 1 / 16
                    
    return dithered

def generate_svg_panel(mode="dark"):
    """Builds the SVG structure with enforced textLength attributes."""
    bg_color = PALETTE["bg"] if mode == "dark" else "#FFFFFF"
    text_color = PALETTE["chrome_dark"] if mode == "dark" else PALETTE["chrome_light"]
    
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}" width="{SVG_WIDTH}" height="{SVG_HEIGHT}">
    <rect width="{SVG_WIDTH}" height="{SVG_HEIGHT}" fill="{bg_color}" />
    <g font-family="monospace" font-size="14" fill="{text_color}">
        <text x="460" y="40" font-size="13" font-weight="bold">profile.sh --live</text>
        <circle cx="1140" cy="35" r="5" fill="red">
            <animate attributeName="opacity" values="1;0.2;1" dur="2s" repeatCount="indefinite"/>
        </circle>
        <text x="1100" y="40" font-size="12" fill="red">LIVE</text>
    '''
    
    y_offset = 100
    for key, value in USER_DATA.items():
        leaders = compute_dotted_leaders(key, value)
        
        # textLength locking for flawless right-alignment regardless of system font
        svg += f'''
        <text x="460" y="{y_offset}">{key}</text>
        <text x="{460 + len(key)*8 + 10}" y="{y_offset}" opacity="0.4">{leaders}</text>
        <text x="1140" y="{y_offset}" text-anchor="end" textLength="{len(value)*8}" lengthAdjust="spacingAndGlyphs">{value}</text>
        '''
        y_offset += 23
        
    svg += '''
    </g>
    <!-- Portrait dots and animation logic will be appended here -->
    </svg>'''
    
    return svg

def build():
    # 1. Ensure you have your portrait image named 'photo.jpg'
    # dithered_dark = process_portrait("photo.jpg", mode="dark")
    
    # 2. Generate Base SVG
    dark_svg = generate_svg_panel("dark")
    light_svg = generate_svg_panel("light")
    
    with open("dark.svg", "w", encoding="utf-8") as f:
        f.write(dark_svg)
    with open("light.svg", "w", encoding="utf-8") as f:
        f.write(light_svg)
        
    print("Base SVGs generated. (Run optimal transport for logos to inject dots).")

if __name__ == "__main__":
    build()
