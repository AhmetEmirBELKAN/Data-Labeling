import os
import cv2

def resize_images_in_folder(folder_path, output_folder, target_size=(100, 100)):
    
    for filename in os.listdir(folder_path):
        
        file_path = os.path.join(folder_path, filename)
        
        
        if os.path.isfile(file_path) and any(file_path.endswith(extension) for extension in ['.jpg', '.png', '.jpeg']):
            
            image = cv2.imread(file_path)
            
            
            resized_image = cv2.resize(image, target_size)
            
            
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, resized_image)
            
            print(f"{filename}  yeniden boyutlandırıldı.")


klasor_yolu = "dogs"
cikti_klasor_yolu = "dogs"
resize_images_in_folder(klasor_yolu, cikti_klasor_yolu)
