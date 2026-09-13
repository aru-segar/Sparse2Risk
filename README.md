# Sparse2Risk

Sparse2Risk is a final-year research project investigating whether sparse current-scan LiDAR supervision can be used to train a post-hoc auditor that predicts dense pixel-wise error risk for a frozen monocular metric-depth model.

## Main idea

The monocular depth teacher receives RGB imagery and predicts metric depth.

During auditor training, sparse current-scan LiDAR provides supervision for teacher-error estimation.

During validation and testing, accumulated reference depth is used to evaluate whether the auditor can predict teacher error at locations that were not observed by the current LiDAR scan.

## Development environment

- Python 3.11
- VS Code
- Git / GitHub
- Google Colab T4 for GPU experiments
- KITTI Depth Completion dataset