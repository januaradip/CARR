# CARR: Customer Activity Recognition in Retail

Code for the CARR dataset: data loading, evaluation protocols, the 71 baseline runs reported in the data
article, and the scripts that produce its tables and figures.

CARR contains **833 trimmed surveillance videos (327,919 annotated frames)** of six customer–shelf activities
performed by **15 participants in 13 operating retail stores** in East Java, Indonesia. Seven participants use a
mobility aid (two wheelchair users, three crutch users and two walking-cane users). Every frame carries the
bounding box of the target participant, the activity class and the participant identifier, so the data supports
detection, activity recognition and person re-identification across videos and stores.

| | |
|---|---|
| Dataset | Harvard Dataverse, [doi:10.7910/DVN/CY24UB](https://doi.org/10.7910/DVN/CY24UB) (first version: Figshare, [doi:10.6084/m9.figshare.29470079](https://doi.org/10.6084/m9.figshare.29470079)) |
| Data article | *Data in Brief* (under review) |
| Code licence | MIT |

## Activity classes

| ID | Class | Videos | Frames |
|---:|---|---:|---:|
| 0 | No Interest | 137 | 46,471 |
| 1 | Viewing | 150 | 54,645 |
| 2 | Turning To Shelf | 136 | 59,499 |
| 3 | Touching | 137 | 60,983 |
| 4 | Picking and Returning | 135 | 66,423 |
| 5 | Picking and Putting | 138 | 39,898 |

## Repository content

```
run_benchmark.py              run the baselines (all or a selection of runs)
carr/
  common.py                   dataset access, splits, run bookkeeping, generic trainer
  detection.py                YOLOv8m, YOLOv12m                          runs R01–R18
  skeleton.py                 LSTM, ST-GCN, CTR-GCN on YOLOv8m-Pose keypoints   R19–R45
  video.py                    R(2+1)D-18, VideoMAE-B                     runs R46–R63
  reid.py                     BoT-ResNet50, ViT-B/16                     runs R64–R71
splits/
  video_splits.csv            train / val / test role of every video in each protocol
  splits_summary.csv          number of videos and frames per subset
  reid/                       re-identification folds (train, query, gallery)
tools/
  make_splits.py              expand video_splits.csv into the frame lists of every protocol
  check_dataset.py            consistency check of a downloaded copy
  dataset_figures.py          Fig. 4 (statistics), Fig. 7 (protocols) and the counts of Tables 1–3, 6
  results_figures.py          Tables 10–13 and Figs. 8–10 from finished runs
notebooks/
  CARR_benchmark_colab.ipynb  the same benchmark on Google Colab
```

## Dataset layout

Download the dataset from Harvard Dataverse, extract `CARR_annotations.zip` and `CARR_splits.zip` into one folder and put the
image archives `Images_carr_part_*.zip` in `images/` (they can stay zipped). The code expects:

```
CARR/
├── carr_manifest.csv          one row per frame: file_name, class_id, class_name, person_id,
│                              participant_code, location_name, video_id, frame_index, from_multi_person_frame
├── images/                    <Class> (<N>).jpg  (a folder of JPEG files or .zip archives of them)
├── labels/                    class_id x_center y_center width height          (YOLO format, normalised)
├── labels_with_person_id/     person_id class_id x_center y_center width height
├── classes_and_persons.yaml
├── cleaning_log.csv
└── splits/                    cs1–cs4, cl1–cl4, cg _{train,val,test}.txt and reid/
```

Image and label files of a frame share the same name. Each label file contains one line, the target participant.
On Google Drive, where hundreds of thousands of small files are slow, a *packed* layout is also accepted:
`dataset/CARR_labels_clean.zip` (manifest and labels), `dataset/CARR_splits.zip` and `dataset/images/*.zip`.

If the `splits/` folder is missing, recreate it from the manifest:

```bash
python tools/make_splits.py --manifest /path/to/CARR/carr_manifest.csv --out /path/to/CARR/splits
python tools/check_dataset.py --data /path/to/CARR
```

## Evaluation protocols

All splits are defined at video level, so frames of one video never occur in more than one subset.
In every protocol 10% of the training videos, stratified by class, are held out for validation.

| Protocol | Test set | Videos (train / val / test) | Test frames |
|---|---|---|---:|
| CS-1 | participants P02, P04, P12, P13 | 654 / 73 / 106 | 35,019 |
| CS-2 | participants P05, P07, P09 | 666 / 75 / 92 | 36,263 |
| CS-3 | participants P00, P08, P10, P14 | 691 / 77 / 65 | 33,931 |
| CS-4 | participants P06, P11 | 685 / 75 / 73 | 31,490 |
| CL-1 | stores Alif, Hafidz 1, KPRI Sumatera | 581 / 65 / 187 | 80,945 |
| CL-2 | stores Haza Mart, Rehana, UNEJ Mart | 582 / 66 / 185 | 80,328 |
| CL-3 | stores Hari, KPRI Kalimantan, Senyum, Senyum 3 | 542 / 58 / 233 | 83,390 |
| CL-4 | stores Budi Jaya, Hafidz 2, Senyum 2 | 545 / 60 / 228 | 83,256 |
| CG | all participants with disabilities (P07–P13) | 686 / 74 / 73 | 46,424 |

* **Cross-subject (CS):** test participants are never seen in training; P01 and P03, who contribute 58% of
  the frames, are always in the training set, and every fold contains participants with and without disabilities.
* **Cross-location (CL):** test stores are never seen in training; the three stores where participants with
  disabilities were recorded are in different folds.
* **Cross-group (CG):** training on participants without disabilities, testing on participants with disabilities.
* **Re-identification (ReID-1 to ReID-4):** 7–8 test identities per fold; for each test identity one video forms
  the query set and the other videos the gallery. Gallery images from the query video are excluded.

## Installation

```bash
git clone https://github.com/januaradip/CARR.git
cd CARR
pip install -r requirements.txt
```

A CUDA GPU is required for training. Pre-trained weights (YOLO, Kinetics-400, ImageNet) are downloaded
automatically on first use.

## Running the baselines

```bash
# list the runs and their status
python run_benchmark.py --data /path/to/CARR --list

# one family, all splits
python run_benchmark.py --data /path/to/CARR --family skeleton

# a selection
python run_benchmark.py --data /path/to/CARR --family video --models VideoMAE-B --splits CS1 CS2 CS3 CS4
python run_benchmark.py --data /path/to/CARR --runs R37 R38 R39 R40
```

Output goes to `--work` (default `./carr_work`):

```
carr_work/
├── cache/        frame index, 640-px image shards, pose keypoints, person crops (built once)
├── runs/<family>/R37_CTR-GCN_CS1/   checkpoints, history.json, confusion_*.json, DONE.json
└── results/      <family>_results.csv (one row per run) and <family>_summary.csv (mean ± std)
```

Runs are resumable: a finished run (with `DONE.json`) is skipped, an interrupted run continues from its last
epoch. Several machines or Colab sessions can share one work folder; each run is locked by the session working on
it, and `--reverse` lets a second session start from the end of the list.

On Google Colab, open `notebooks/CARR_benchmark_colab.ipynb`. To continue runs that were started with the
packed layout on Drive, use the dataset folder itself as `WORK`.

### Baseline settings

| Family | Models | Input | Training |
|---|---|---|---|
| Detection | YOLOv8m, YOLOv12m (COCO pre-trained) | 640 px; every 10th training frame | ≤ 50 epochs, early stop 10, batch 16; evaluated on all test frames |
| Skeleton | LSTM, ST-GCN, CTR-GCN | 17 COCO keypoints (YOLOv8m-Pose) normalised by box centre and height; windows of 64 frames, stride 32 | AdamW, lr 1e-3, ≤ 60 epochs, early stop 12, batch 64 |
| Video | R(2+1)D-18, VideoMAE-B (Kinetics-400) | person crops; 16-frame clips covering 64 source frames; 112 / 224 px | AdamW, lr 2e-4 / 5e-5, ≤ 20 / 15 epochs, early stop 5 |
| Re-ID | BoT-ResNet50, ViT-B/16 (ImageNet) | 256 × 128 crops | ID + batch-hard triplet loss, BNNeck, 40 epochs, P × K = 7 × 8 |

Activity recognition is reported as macro-F1 at window level and at video level (class probabilities of all
windows of a video averaged), detection as mAP@50, mAP@50:95, precision and recall, and re-identification as
Rank-1, Rank-5 and mAP, all as mean ± standard deviation over folds. All runs use seed 0; exact values depend on
the GPU and library versions. The settings are defined in the `CFG` dictionary at the top of each module.

## Reproducing the tables and figures

```bash
python tools/dataset_figures.py --data /path/to/CARR --out figures      # Fig. 4, Fig. 7, counts of Tables 1-3, 6
python tools/results_figures.py --work carr_work --out figures          # Tables 10-13, Figs. 8-10
```

## Privacy

Participants are identified only by the codes P00–P14. File names contain the activity class and a running
number, and do not identify participants or stores. Faces are blurred in the figures of the article. Use of the
data must respect the licence stated on the Harvard Dataverse dataset page; do not attempt to identify the participants.

## Citation

If you use CARR, please cite the dataset and the data article:

```bibtex
@misc{putra2026carr_data,
  author    = {Putra, Januar Adi and Suciati, Nanik and Fatichah, Chastine and Nabil, Fabian and Ayuningtyas, Tiara Salsabella},
  title     = {Customer Activity Recognition in Retail ({CARR}) dataset},
  publisher = {Harvard Dataverse},
  year      = {2026},
  doi       = {10.7910/DVN/CY24UB}
}
```

The reference of the data article will be added after publication.
