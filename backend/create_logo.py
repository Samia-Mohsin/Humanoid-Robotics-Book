from PIL import Image, ImageDraw
import os

# Create a hexagon-shaped logo with robot and brain elements
def create_logo():
    # Create a 512x512 image with transparent background
    img = Image.new('RGBA', (512, 512), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)

    # Draw a hexagon frame
    hexagon_points = [
        (256, 50),    # Top
        (450, 150),   # Top-right
        (450, 362),   # Bottom-right
        (256, 462),   # Bottom
        (62, 362),    # Bottom-left
        (62, 150)     # Top-left
    ]

    # Draw the hexagon with a blue-green gradient
    draw.polygon(hexagon_points, outline=(0, 200, 255), width=10, fill=(0, 30, 60, 180))

    # Draw a simple robot head shape
    draw.ellipse((180, 180, 332, 300), outline=(0, 255, 255), width=5, fill=(0, 40, 80, 200))

    # Draw a brain shape inside the robot head
    brain_points = [
        (220, 220),
        (230, 210), (240, 215), (250, 210), (260, 220),
        (270, 225), (260, 230), (250, 235), (240, 230), (230, 235), (220, 240)
    ]
    draw.polygon(brain_points, outline=(0, 255, 150), width=3, fill=(0, 80, 100, 220))

    # Draw a light bulb above the robot
    draw.ellipse((236, 140, 276, 180), outline=(255, 255, 0), width=3, fill=(255, 255, 200, 220))
    draw.rectangle((251, 180, 261, 200), outline=(255, 255, 0), width=3, fill=(255, 255, 200, 220))

    # Save as PNG
    img.save('static/img/logo.png')
    print("Logo created successfully at static/img/logo.png")

if __name__ == "__main__":
    create_logo()