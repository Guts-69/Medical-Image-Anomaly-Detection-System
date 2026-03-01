import os

DATASET_PATH = r"C:\Users\Aditya\Desktop\Medical_Image_Anomaly_Detection\data"  # adjust if needed

for split in os.listdir(r"C:\Users\Aditya\Desktop\Medical_Image_Anomaly_Detection\data"):
    split_path = os.path.join(r"C:\Users\Aditya\Desktop\Medical_Image_Anomaly_Detection\data", split)
    if os.path.isdir(split_path):
        print(f"\n--- {split} ---")
        
        for class_name in os.listdir(split_path):
            class_path = os.path.join(split_path, class_name)
            if os.path.isdir(class_path):
                num_images = len(os.listdir(class_path))
                print(f"{class_name}: {num_images} images")
