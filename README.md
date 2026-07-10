# Korea Wildfire Spread Prediction U-Net

한국 산불 확산 예측을 위한 U-Net 기반 딥러닝 연구 코드입니다.  
본 연구는 **Day T의 산불 화점, 기상, 지형, 접근성 정보**를 입력으로 사용하여 **Day T+1의 산불 화점 확률 지도**를 `64×64` 격자로 예측하는 것을 목표로 합니다.

이 저장소는 한국 산악지형과 소방 접근성 정보를 반영한 **14채널 terrain + accessibility U-Net 모델** 학습, 특성중요도 분석, ablation study를 수행하기 위한 코드로 구성되어 있습니다.

---

## Research Overview

입력 데이터는 한 지역의 하루 단위 정보를 `14 × 64 × 64` tensor로 구성합니다. 출력은 다음 날 산불 화점 여부를 나타내는 `1 × 64 × 64` 확률 지도입니다.

### Input Features

| Channel | Feature | Description |
|---:|---|---|
| 0 | `fire_mask_t` | Day T 산불 화점 마스크 |
| 1 | `temperature` | 기온 |
| 2 | `humidity` | 습도 |
| 3 | `u_wind` | 동서 방향 바람 성분 |
| 4 | `v_wind` | 남북 방향 바람 성분 |
| 5 | `dem_elevation` | DEM 고도 |
| 6 | `slope_deg` | 경사도 |
| 7 | `aspect_sin` | 사면 방향 sin 값 |
| 8 | `aspect_cos` | 사면 방향 cos 값 |
| 9 | `tpi_5x5` | 지형 위치 지수 |
| 10 | `relative_elevation` | 주변 대비 상대 고도 |
| 11 | `roughness_5x5` | 지형 거칠기 |
| 12 | `wind_slope_alignment` | 바람-사면 정렬도 |
| 13 | `nearest_fire_station_distance` | 가장 가까운 소방시설까지 거리 |

### Output

```text
Day T+1 wildfire probability map
shape: (1, 64, 64)
```

---

## Dataset

학습 데이터는 NASA FIRMS/VIIRS 화점, 기상청 ASOS 기상자료, SRTM DEM 지형자료, 소방시설 위치 정보를 결합하여 구성했습니다.

```text
data/
  X_train_terrain14_accessibility_2021_2024.npy
  Y_train_2021_2024.npy
  sample_index_2021_2024.csv
  results/
```

예상 데이터 shape는 다음과 같습니다.

```text
X: (13140, 14, 64, 64)
Y: (13140, 1, 64, 64)
```

`sample_index_2021_2024.csv`에는 `train`, `val`, `test` split 정보가 포함됩니다.  
학습, threshold 선택, feature importance, ablation은 validation set 기준으로 수행하고, test set은 최종 평가 단계에서만 사용합니다.

---

## Main Files

| File | Description |
|---|---|
| `model.py` | U-Net 구조, SEBlock, Focal Loss, Dice Loss, CombinedLoss 정의 파일입니다. |
| `model_14.py` | 14채널 terrain + accessibility U-Net 기본 학습 스크립트입니다. 본 연구의 메인 학습 코드입니다. |
| `permutation_importance_14.py` | 학습된 14채널 모델에 대해 validation set 기준 permutation importance를 계산합니다. |
| `model_ablation_resume.py` | 특정 feature를 입력에서 제외하고 U-Net을 재학습하는 ablation study 스크립트입니다. |
| `permutation_importance_selected.py` | ablation 모델 또는 선택된 feature 조합 모델의 permutation importance를 다시 계산합니다. |
| `check_data.py` | Docker 컨테이너 안에서 데이터 파일과 shape가 올바른지 확인합니다. |
| `Dockerfile` | PyTorch + CUDA 기반 Docker 실행환경을 정의합니다. |
| `requirements.txt` | Docker 이미지 내부에서 추가 설치할 Python 패키지 목록입니다. |
| `.dockerignore` | 데이터, checkpoint, 결과 파일이 Docker image에 포함되지 않도록 제외합니다. |
| `baseline_us_fixed.py` | Huot et al.의 미국 Next Day Wildfire Spread baseline 재현용 코드입니다. 한국 14채널 U-Net의 메인 코드는 아닙니다. |

---


## Training

14채널 U-Net 학습 실행 예시입니다.

```bash
docker run --rm --gpus all --ipc=host \
  -v "$PWD/data:/workspace/data" \
  wildfire-unet:model14 \
  python -u model_14.py --data_dir /workspace/data --epochs 30 --batch_size 32
```

학습 후 주요 출력 파일은 다음과 같습니다.

```text
data/unet_terrain14_accessibility_best.pth
data/terrain14_accessibility_train_history.csv
```

---

## Feature Importance and Ablation

14채널 모델의 특성중요도는 validation set에서 특정 feature 채널을 섞었을 때 AUC-PR이 얼마나 감소하는지로 계산합니다.

```bash
python -u permutation_importance_14.py \
  --data_dir /workspace/data \
  --batch_size 64 \
  --num_workers 4
```

Ablation study는 원본 `.npy` 파일을 수정하지 않고, 코드에서 특정 feature 채널만 제외한 뒤 모델을 다시 학습하는 방식으로 진행합니다.

```bash
python -u model_ablation_resume.py \
  --data_dir /workspace/data \
  --run_name ablate_1_drop_temp \
  --drop_features temperature \
  --epochs 30 \
  --batch_size 32 \
  --num_workers 4
```

---

## Notes

- `model_14.py`가 기본 학습 스크립트입니다.
- feature 선택과 threshold 선택은 validation set 기준으로 수행합니다.
- test set은 최종 평가 전까지 사용하지 않습니다.
- `duration_hours`, `duration_days`, `burned_area_ha`처럼 산불 종료 후에만 알 수 있는 사후 정보는 모델 입력으로 사용하지 않습니다.
