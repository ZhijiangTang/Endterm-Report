
[CVPR Papers](https://github.com/amusi/CVPR2025-Papers-with-Code/tree/main)

```bash
ros2 bag play ./data/pku_campus.db3 
python3 CompressedToImageNode.py
# DEIM_S RTDETR_S YOLO image
python3 run_ros2.py --model_name DEIM_S
python3 run_ros2.py --model_name RTDETR_S
python3 run_ros2.py --model_name YOLO
# -p image_save_path:="./data/image/"
ros2 run rviz2 rviz2
ros2 topic echo /front/compressed --no-arr
```