# HandPD Parkinson's Detection

This project trains a dual-branch CNN using paired Spiral and Meander handwriting images from the HandPD dataset.

## Project Structure

```text
handpd_5th_newHandPd_dataset/
├── NewHandPD Dataset/     # Training dataset and CSV files
├── notebooks/             # Training notebook
├── models/                # Saved trained models
├── input_images/          # New Spiral/Meander image pairs
├── predict.py             # Prediction script
└── README.MD
```

## Environment Setup

Open PowerShell in the project folder:

```powershell
python -m venv .venv

cd "C:\Users\kanis\Desktop\model\handpd_5th_newHandPd_dataset"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Check the required packages:

```powershell
python -c "import tensorflow, sklearn, pandas, numpy, PIL; print('Packages OK')"
```

If packages are missing:

```powershell
python -m pip install tensorflow scikit-learn pandas numpy matplotlib pillow
```

## Train the Model

Open the notebook:

```powershell
code ".\notebooks\Parkinsons_Dual_CNN_clean (1).ipynb"
```

In VS Code:

1. Select the `.venv` Python kernel.
2. Run the notebook cells from top to bottom.
3. Run the data preparation cells.
4. Run the model and 5-fold cross-validation training cell.
5. Run the threshold and test evaluation cells.
6. Run the final save cell.

Training creates these files in `models/`:

```text
fold1_192px.keras
fold2_192px.keras
fold3_192px.keras
fold4_192px.keras
fold5_192px.keras
config_192px.json
```

Training is required only when the dataset or training code changes. Do not retrain for every new prediction.

## Predict a New Image Pair

Place a Spiral and Meander image in `input_images/`. The two images must belong to the same person and examination.

Run:

```powershell
python predict.py --spiral input_images\spiral_new.jpg --meander input_images\meander_new.jpg
```

The script loads all five saved models, preprocesses both images in the same way as training, averages the predictions, and prints the Parkinson's probability and result.

## Important Limitations

- The model requires both a Spiral image and a Meander image.
- Images should use a format similar to the training dataset.
- The current dataset is small, so false positives and false negatives are possible.
- The model is an experimental research model and must not be used as a medical diagnosis.
- Validate new model versions using unseen healthy and Parkinson's participants before using them in another application.
