# Multi-thread SOT 

Built on a combination of SiamRPN, Detectron2, and deep-person-reid. 

# Usage

Sample usage: 

```python eval_folder.py --data_dir ../dataset/dataset/LaSOT/car-14/img --config config.
yaml --init_rect 614,389,103,99 --output_video result.avi 
```

```python eval_folder.py --data_dir <video or image folder directory> --config config.
yaml --init_rect <x,y,w,h> --output_video result.avi 
```