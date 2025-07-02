# Customer Activity Recognition in Retail (CARR) Dataset

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](http://creativecommons.org/licenses/by/4.0/)

## Overview

The **Customer Activity Recognition in Retail (CARR)** dataset is a publicly available dataset designed to support research and development in **customer behavior analysis** using video surveillance. The dataset includes annotated videos and frames of customer activities captured in realistic retail environments using a single surveillance camera with a tilted angle of 45–90°.

This dataset is intended for training and evaluating machine learning and deep learning models—particularly object detection, pose estimation, and activity classification—in retail contexts.

---

## Dataset Summary

- **Number of Activities:** 6 classes  
  - `No Interest`
  - `Viewing`
  - `Turning to Shelf`
  - `Touching`
  - `Picking and Returning`
  - `Picking and Putting`

- **Number of Videos:** 743  
- **Participants:** 7 diverse individuals (gender, age, height, weight)  
- **Recording Environment:** 10 real retail locations, uncontrolled settings  
- **Camera Setup:** EZVIZ H80x Dual 4K, 2–3m high, 45–90° tilt  
- **Formats:** `.mp4`, `.png`, `.txt` (YOLO annotation format)

---

## Data Structure

After extraction, the dataset is organized as:

```
CARR/
├── Retail_Location_01/
│   ├── no_interest/
│   ├── viewing/
│   ├── turning_to_shelf/
│   ├── touching/
│   ├── picking_and_returning/
│   └── picking_and_putting/
├── Retail_Location_02/
...
├── CARR_labels/
└── README.md
```

Each class folder contains:
- Video clips (`.mp4`)
- Extracted frames (`.png`)
- Annotation files (`.txt`) following YOLO format:
  ```
  <class_id> <x_center> <y_center> <width> <height>
  ```

---

## Annotation Details

Annotations were created manually using **Label Studio**. Each bounding box is associated with a customer activity class. Class IDs:

| Class               | ID |
|---------------------|----|
| No Interest         | 0  |
| Viewing             | 1  |
| Turning to Shelf    | 2  |
| Touching            | 3  |
| Picking and Returning | 4  |
| Picking and Putting | 5  |

---

## Benchmark & Evaluation

### Object Detection with YOLOv8–v12
### Pose Estimation + Classifier

More detailed evaluation and tables can be found in the article

---

## Usage

You may use this dataset to:

- Develop and evaluate customer activity recognition algorithms
- Test pose estimation or multi-class action classification models
- Simulate realistic retail behavior analysis scenarios

---

## Acknowledgements

This work was supported by the **Indonesian Endowment Fund for Education (LPDP)**. We thank all participants who contributed to the dataset creation.

---

## License

This dataset is licensed under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

---

## Contact

For questions or collaborations, please contact:  
**Dr. Nanik Suciati**  
Informatics Department, ITS  
Email: nanik@its.ac.id
