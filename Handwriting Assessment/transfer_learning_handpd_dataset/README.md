# HandPD EfficientNet Model Testing

This README explains how to run the saved HandPD EfficientNetB0 transfer-learning model from the beginning in PowerShell.

## 1. Open PowerShell and go to the project

```powershell
cd "C:\Users\kanis\Desktop\model\transfer learning"
```

The project folder should contain:

```text
test_model.py
model\fold1.keras ... fold5.keras
model\config.json
data\handpd_dataset\
```

## 2. Create and activate the Python environment

Run these commands once:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

The PowerShell prompt should now start with `(.venv)`.

If `.venv` already exists, run only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

## 3. Install required packages

```powershell
python -m pip install --upgrade pip
python -m pip install tensorflow numpy scikit-learn pillow
```

## 4. Check the command options

```powershell
python .\test_model.py --help
```

## 5. Quick model test

This loads all five saved fold models and tests two dataset pairs:

```powershell
python .\test_model.py --limit 2
```

## 6. Test the complete dataset

```powershell
python .\test_model.py
```

The output shows:

- Raw probability AUC
- Normalized score AUC
- Confusion matrix
- Classification report
- Balanced accuracy
- Macro F1 score

## 7. Test one Spiral and Meander image

The Meander filename must use the Spiral number plus 4. For example, this is a valid pair from the dataset:

```powershell
python .\test_model.py `
  --spiral ".\data\handpd_dataset\SpiralControl\0068-1.jpg" `
  --meander ".\data\handpd_dataset\MeanderControl\0068-5.jpg"
```

The output contains the ensemble probability, normalized score, and final `Control` or `Patient` prediction.

## 8. Find image filenames

If you want to test another pair, list the available files:

```powershell
Get-ChildItem .\data\handpd_dataset\SpiralControl -File
Get-ChildItem .\data\handpd_dataset\MeanderControl -File
Get-ChildItem .\data\handpd_dataset\SpiralPatients -File
Get-ChildItem .\data\handpd_dataset\MeanderPatients -File
```

Pairing examples:

```text
SpiralControl\0068-1.jpg  -> MeanderControl\0068-5.jpg
SpiralPatients\0123-2.jpg -> MeanderPatients\0123-6.jpg
```

## 9. If memory is limited

Use a smaller batch size:

```powershell
python .\test_model.py --batch-size 4
```

## 10. Stop the environment

```powershell
deactivate
```

TensorFlow startup warnings about CPU instructions or oneDNN are normal. The important result is the final prediction or metric output.
