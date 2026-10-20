# HandPD Spiral + Meander Fusion Model

This project trains a TensorFlow/Keras model to classify handwriting samples as Parkinson's patients or controls using Spiral and Meander images.

## 1. Set Up the Environment

Open PowerShell in the project root folder and activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the required packages if they are not already installed:

```powershell
python -m pip install numpy pandas scipy scikit-learn matplotlib tensorflow
```

The dataset zip file must be in the project root:

```text
handpd_dataset.zip
```

## 2. Train the Model

```powershell
python .\train_handpd.py train
```

Training uses subject-level 5-fold cross-validation. After training, the models are saved here:

```text
model\fold1.keras
model\fold2.keras
model\fold3.keras
model\fold4.keras
model\fold5.keras
model\config.json
```

The `model\config.json` file stores the image size and the threshold for each fold.

Training may take a long time when running on a CPU.

## 3. Test the Saved Model

Test a patient sample:

```powershell
python .\train_handpd.py test --spiral ".\Data\handpd_dataset\SpiralPatients\0002-1.jpg" --meander ".\Data\handpd_dataset\MeanderPatients\0002-5.jpg"
```

Test a control sample:

```powershell
python .\train_handpd.py test --spiral ".\Data\handpd_dataset\SpiralControl\0068-1.jpg" --meander ".\Data\handpd_dataset\MeanderControl\0068-5.jpg"
```

Example output:

```json
{
  "score": 4.475,
  "prediction": "Patient"
}
```

The `prediction` value will be either `Patient` or `Control`.

## 4. Model Quality

The evaluation uses a subject-level split, so images from the same subject are not placed in both the training and test groups. This reduces the risk of data leakage.

Because fold results can vary, review the mean AUC, individual fold AUC values, confusion matrix, and classification report together. More data and external validation are required before using this model for medical diagnosis.

## 5. Project Files

```text
train_handpd.py       # Training and testing commands
handpd_dataset.zip    # Input dataset
Data/                 # Extracted dataset
model/                # Saved fold models and configuration
notebook/             # Jupyter notebooks
```

---

# සිංහල උපදෙස්

මෙම project එක Spiral සහ Meander handwriting images භාවිතා කරලා Parkinson's patient ද control ද කියලා classify කරන TensorFlow/Keras model එකක් train කරයි.

## 1. Environment එක සකස් කිරීම

Project root folder එකේ PowerShell එකක් open කරලා:

```powershell
.\.venv\Scripts\Activate.ps1
```

අවශ්‍ය packages නැත්නම්:

```powershell
python -m pip install numpy pandas scipy scikit-learn matplotlib tensorflow
```

Dataset zip file එක project root එකේ තිබිය යුතුයි:

```text
handpd_dataset.zip
```

## 2. Model එක train කිරීම

```powershell
python .\train_handpd.py train
```

Training එකෙන් subject-level 5-fold cross-validation සිදු වේ. Training අවසන් වූ පසු models මෙතන save වේ:

```text
model\fold1.keras
model\fold2.keras
model\fold3.keras
model\fold4.keras
model\fold5.keras
model\config.json
```

`model\config.json` file එකේ image size සහ එක් එක් fold එකේ threshold save වේ.

CPU එකෙන් run කළොත් training එකට දිගු වේලාවක් ගත විය හැක.

## 3. Saved model එක test කිරීම

Patient sample එකක් test කිරීම:

```powershell
python .\train_handpd.py test --spiral ".\Data\handpd_dataset\SpiralPatients\0002-1.jpg" --meander ".\Data\handpd_dataset\MeanderPatients\0002-5.jpg"
```

Control sample එකක් test කිරීම:

```powershell
python .\train_handpd.py test --spiral ".\Data\handpd_dataset\SpiralControl\0068-1.jpg" --meander ".\Data\handpd_dataset\MeanderControl\0068-5.jpg"
```

Output එක මෙවැනි ආකාරයක වේ:

```json
{
  "score": 4.475,
  "prediction": "Patient"
}
```

`prediction` අගය `Patient` හෝ `Control` වේ.

## 4. Model quality

Evaluation එකේදී subject එකේ images train සහ test groups දෙකට බෙදී නොයන ලෙස subject-level split එකක් භාවිතා කරයි. එම නිසා data leakage අවදානම අඩු වේ.

කෙසේ වෙතත්, fold results වෙනස් විය හැකි නිසා mean AUC, individual fold AUC, confusion matrix, සහ classification report එක එකට බලන්න. මෙය medical diagnosis සඳහා භාවිතා කිරීමට පෙර වැඩි data සහ external validation අවශ්‍ය වේ.

## 5. Files

```text
train_handpd.py       # Train සහ test commands
handpd_dataset.zip    # Input dataset
Data/                 # Extract කළ dataset
model/                # Saved fold models සහ config
notebook/             # Jupyter notebooks
```
