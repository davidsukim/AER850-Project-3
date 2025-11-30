from ultralytics import YOLO
import torch
import os
import time
import datetime

def train_pcb_model():
    # 1. Check if GPU is availiable
    if torch.cuda.is_available():
        print(f"GPU found: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU NOT found")

    # 2. dataset location
    dataset_yaml = r"D:\My Ryerson\7. Seventh Semester\AER 850\Project\3\Project 3 Data\Project 3 Data\data\data.yaml"
    
    if not os.path.exists(dataset_yaml):
        print("'data.yaml' NOT FOUND. CHECK THE FILE LOCATION")
        return

    # 3. loading the model (Brining the light nano model)
    model = YOLO('yolo11n.pt') 

    # 4. ML START!!!
    print(f"🚀 Machine Learning Started... (Current Time: {datetime.datetime.now().strftime('%H:%M:%S')})")
    start_time = time.time()  # recording the time took
    
    results = model.train(
        data=dataset_yaml,
        epochs=150,           
        imgsz=960,
        batch=4,              
        device=0,             
        name='pcb_yolo_run',
        plots=True,
        workers=0       # Prevent Error
    )

    # 5. Finishing Time Calculation
    end_time = time.time()
    elapsed_time = end_time - start_time 
    elapsed_str = str(datetime.timedelta(seconds=int(elapsed_time)))

    print("\n" + "="*40)
    print(f"ML successfully finished!! Time: {elapsed_str}")
    print(f"   (Result Folder: runs/detect/pcb_yolo_run)")
    print("="*40)

if __name__ == '__main__':
    train_pcb_model()