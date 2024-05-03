import cv2
from env import CAPTURE_TIME_BUFFER, CAPTURE_EVERY_X_FRAMES


def resize_image(image, max_width=500):
    """
    Resizes and returns the provided image with the maximum width
    """
    # Get the dimensions of the image
    height, width = image.shape[:2]

    # Calculate the ratio of the width and apply it to the new width
    ratio = max_width / float(width)
    new_height = int(height * ratio)

    # Resize the image
    resized_image = cv2.resize(
        image, (max_width, new_height), interpolation=cv2.INTER_AREA
    )
    return resized_image


def add_subtitle(image, frame_count, text="", show_countdown = True, max_line_length=40):
    """
    Adds subtitle text to the image at the bottom and inserts a countdown depending on the frame count
    """
    countdown_seconds = CAPTURE_TIME_BUFFER - (frame_count % CAPTURE_EVERY_X_FRAMES) // 30  # Convert frame count to seconds

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)  # White color for the main text
    shadow_color = (0, 0, 0)  # Black color for the shadow
    line_type = 2
    margin = 10  # Margin for text from the bottom
    line_spacing = 30  # Space between lines
    shadow_offset = 2  # Offset for shadow

    # Split text into multiple lines
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        if len(current_line + word) <= max_line_length:
            current_line += word + " "
        else:
            lines.append(current_line)
            current_line = word + " "
    lines.append(current_line)  # Add the last line

    # Calculate the starting y position
    text_height_total = line_spacing * len(lines)
    start_y = image.shape[0] - text_height_total - margin

    for i, line in enumerate(lines):
        text_size = cv2.getTextSize(line, font, font_scale, line_type)[0]
        text_x = (image.shape[1] - text_size[0]) // 2
        text_y = start_y + i * line_spacing

        # Draw shadow
        cv2.putText(
            image,
            line,
            (text_x + shadow_offset, text_y + shadow_offset),
            font,
            font_scale,
            shadow_color,
            line_type,
        )

        # Draw main text
        cv2.putText(
            image, line, (text_x, text_y), font, font_scale, font_color, line_type
        )

    font_scale = 2
    countdown_text = f"{countdown_seconds}"
    text_size = cv2.getTextSize(countdown_text, font, font_scale, line_type)[0]
    text_x = image.shape[1] - text_size[0] - margin  # Position to the right
    text_y = margin + text_size[1]  # Position at the top

    if show_countdown:
        # Draw shadow for countdown
        cv2.putText(image, countdown_text, (text_x + shadow_offset, text_y + shadow_offset), font, font_scale, shadow_color, line_type)

        # Draw countdown text
        cv2.putText(image, countdown_text, (text_x, text_y), font, font_scale, font_color, line_type)

    return image