import os
import shutil

# Directories
source_dir = "C:/Coding/Earendel/train"  # Replace with your source folder path
image_target_dir = "C:/Coding/Earendel/keke"  # Replace with your image target folder path
text_target_dir = "C:/Coding/Earendel/hoho"  # Replace with your text target folder path
suffix_to_add = "b80"  # The string to add to the filename

# List of base names
CLASSES = [
    "kek_lapis", "Kuih_Bahulu", "kuih_kaswi_pandan", "Kuih_Ketayap", 
    "Kuih_Lapis", "Kuih_Seri_Muka", "Kuih_Talam", "Kuih_Ubi_Kayu", "Onde_Onde"
]

# Ensure target directories exist
os.makedirs(image_target_dir, exist_ok=True)
os.makedirs(text_target_dir, exist_ok=True)

# Supported image extensions
image_extensions = (".png", ".jpg", ".jpeg")

# Iterate through files in the source directory
for filename in os.listdir(source_dir):
    # Check for image files
    if filename.lower().endswith(image_extensions):
        # Check if the filename starts with any of the CLASSES
        for class_name in CLASSES:
            if filename.startswith(class_name + "_"):
                # Extract base name and number with extension
                base_name = filename.rsplit("_", 1)[0]  # e.g., "kek_lapis"
                number_ext = filename.rsplit("_", 1)[1]  # e.g., "001.png" or "001.jpg"
                
                # Construct new filename
                new_filename = f"{base_name}_{suffix_to_add}_{number_ext}"  # e.g., "kek_lapis_60c_001.png"
                
                # Paths for source and destination
                source_image_path = os.path.join(source_dir, filename)
                target_image_path = os.path.join(image_target_dir, new_filename)
                
                # Move and rename image file
                shutil.move(source_image_path, target_image_path)
                print(f"Moved and renamed image: {filename} -> {new_filename}")
                
                # Check for corresponding text file
                text_filename = filename.rsplit(".", 1)[0] + ".txt"  # e.g., "kek_lapis_001.txt"
                source_text_path = os.path.join(source_dir, text_filename)
                
                if os.path.exists(source_text_path):
                    # Construct new text filename
                    new_text_filename = new_filename.rsplit(".", 1)[0] + ".txt"  # e.g., "kek_lapis_60c_001.txt"
                    target_text_path = os.path.join(text_target_dir, new_text_filename)
                    
                    # Move and rename text file
                    shutil.move(source_text_path, target_text_path)
                    print(f"Moved and renamed text: {text_filename} -> {new_text_filename}")
                else:
                    print(f"No corresponding text file found for: {filename}")
                break  # Exit the class_name loop once matched