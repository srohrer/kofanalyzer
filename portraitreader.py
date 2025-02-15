import cv2
from fastai.vision.all import *

def show_cropped_region(image, roi):
    """Loads an image, crops the given ROI, and saves it for debugging."""
    # Extract ROI
    x, y, w, h = roi
    cropped = image[y:y+h, x:x+w]

    # Save the cropped image instead of showing it
    debug_filename = f"debug_crop_{x}_{y}.png"
    cv2.imwrite(debug_filename, cropped)

def extract_character_names(image, learn):
    """
    Extracts character names from two specific regions of an image using a fastai model.

    :param image_path: Path to the image file.
    :return: A tuple of (player1_char_name, player2_char_name)
    """
    roi_p1 = (0, 0, 50, 30)   # Adjust as needed
    roi_p2 = (590, 0, 50, 30)  # Adjust as needed

    show_cropped_region(image, roi_p1)
    show_cropped_region(image, roi_p2)

    # Process ROI for player 1
    x, y, w, h = roi_p1
    cropped_p1 = image[y:y+h, x:x+w]
    # Convert from BGR to RGB and to PIL Image for fastai
    cropped_p1_rgb = cv2.cvtColor(cropped_p1, cv2.COLOR_BGR2RGB)
    p1_img = PILImage.create(cropped_p1_rgb)
    p1_result = learn.predict(p1_img)[0]  # Get the predicted class name

    # Process ROI for player 2
    x, y, w, h = roi_p2
    cropped_p2 = image[y:y+h, x:x+w]
    cropped_p2_rgb = cv2.cvtColor(cropped_p2, cv2.COLOR_BGR2RGB)
    # Flip the image horizontally
    cropped_p2_rgb = cv2.flip(cropped_p2_rgb, 1)
    p2_img = PILImage.create(cropped_p2_rgb)
    p2_result = learn.predict(p2_img)[0]  # Get the predicted class name

    return str(p1_result), str(p2_result)

# Test the function
if __name__ == "__main__":
    image_path = "kofscreenshot.jpg"
    image = cv2.imread(image_path)
    learn = load_learner('ig_portraits.pkl')
    char_name_p1, char_name_p2 = extract_character_names(image, learn)

    print(f"Player 1 Character: {char_name_p1}")
    print(f"Player 2 Character: {char_name_p2}")
