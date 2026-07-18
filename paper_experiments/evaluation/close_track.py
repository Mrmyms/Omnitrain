from PIL import Image, ImageDraw

def close_map(path):
    img = Image.open(path)
    draw = ImageDraw.Draw(img)
    width, height = img.size
    border_width = 10
    
    # In ROS maps, black (0) is obstacle/wall. 
    # Let's draw a black rectangle border.
    draw.rectangle([(0, 0), (width, border_width)], fill="black") # Top
    draw.rectangle([(0, height - border_width), (width, height)], fill="black") # Bottom
    draw.rectangle([(0, 0), (border_width, height)], fill="black") # Left
    draw.rectangle([(width - border_width, 0), (width, height)], fill="black") # Right
    
    img.save(path)
    print(f"Closed track boundaries for {path}")

close_map("../data/maps/maps/vegas.png")
