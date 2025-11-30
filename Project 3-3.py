from ultralytics import YOLO
import os
import glob
import matplotlib.pyplot as plt
import cv2

def evaluate_project3():
    # 1. Loading the Model
    model_path = r"D:\My Ryerson\7. Seventh Semester\AER 850\Project\3\runs\detect\pcb_yolo_run\weights\best.pt"
    
    if not os.path.exists(model_path):
        print(f"CANNOT FIND THE MODEL: {model_path}")
        return
    
    print(f"Model Loaded: {model_path}")
    model = YOLO(model_path)
    
    # 2. Setting Evaluation Image
    eval_dir = r"D:\My Ryerson\7. Seventh Semester\AER 850\Project\3\Project 3 Data\Project 3 Data\data\evaluation"
    
    # Find all files in the folder
    image_files = glob.glob(os.path.join(eval_dir, "*.*"))
    
    if not image_files:
        print(f"CANNOT FIND THE IMAGE: {eval_dir}")
        return
    
    print(f"Evaluation image found {len(image_files)} Prediction Started...\n")
    
    # 3. Predict and Visualize
    for img_file in image_files:
        
        # Predicting
        results = model.predict(img_file, save=True, imgsz=960, conf=0.25)
                
        # Visualizing
        result_img = results[0].plot()
            
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Prediction: {os.path.basename(img_file)}")
        plt.axis('off')
        plt.show()
        
    print("\n Evaluation Finished!")
    print("   Image Results are saved in 'runs/detect/predict...' Folder")

if __name__ == '__main__':
    evaluate_project3()