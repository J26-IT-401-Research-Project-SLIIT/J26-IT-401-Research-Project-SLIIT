## How to Run

Run these commands in the VS Code terminal (PowerShell).

### 1. Go to the project folder

```powershell
cd C:\Users\kanis\Desktop\model\using_vs_code
```

### 2. Activate the virtual environment

```powershell
python -m venv .venv

.\.venv\Scripts\Activate.ps1
```

If you get a "running scripts is disabled" error, run this first (it only affects the current terminal) and then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

`(.venv)` should now appear at the start of the prompt.

### 3. Go to the notebooks folder

```powershell
cd notebooks
```

### 4. Set the Python path and dataset path

The `.venv` numpy is blocked by Windows Application Control, so Python 3.14 is used to run the backend.

```powershell
$PY = "C:\Users\kanis\AppData\Local\Programs\Python\Python314\python.exe"
$D  = "C:\Users\kanis\Desktop\model\using_vs_code\dataset"
```

### 5. List example image file names (optional)

```powershell
Get-ChildItem "$D\Spiral_HandPD\SpiralControl" | Select-Object -First 5 Name
Get-ChildItem "$D\Meander\MenderControl" | Select-Object -First 5 Name
```

### 6. Run a prediction

Both images must be from the same person.

Patient example:

```powershell
& $PY pd_fusion.py --spiral "$D\Spiral_HandPD\SpiralPatients\0002-1.jpg" --mender "$D\Meander\MenderPatients\0002-5.jpg"
```

Healthy example (replace the file names with real ones from step 5):

```powershell
& $PY pd_fusion.py --spiral "$D\Spiral_HandPD\SpiralControl\<FILE_NAME>.jpg" --mender "$D\Meander\MenderControl\<FILE_NAME>.jpg"
```

The last line of the output shows the result:

```
==> Handwriting PD score: xx.x %  (Parkinson (PD) / Healthy)
```

### 7. (Optional) Run the API server

Install the packages (first time only):

```powershell
& $PY -m pip install fastapi uvicorn python-multipart
```

Start the server:

```powershell
& $PY -m uvicorn pd_fusion:app --reload
```

Open `http://127.0.0.1:8000/docs` in a browser, expand `POST /predict`, click **Try it out**, choose the spiral and mender images, and click **Execute**. Press `Ctrl + C` in the terminal to stop the server.

### 8. Deactivate the virtual environment when finished

```powershell
deactivate
```

**Notes**

- Repeat steps 1–4 every time you open a new terminal (`$PY` and `$D` are lost when the terminal closes).
- This is a research prototype and not a medical diagnosis tool.v