# 🛠️ ChicCheck Model  

🚀 **Smart Self Attendance Check With Liveness Detection** – The backend model that performs liveness detection using **FastAPI** to deploy the model. The model is built to prevent fake attendance attempts by verifying real people.

## 🚀 Getting Started  

### 1️⃣ Clone the repository  
To get started, clone the **chiccheck-model** repository to your local machine:  
```bash
git clone https://github.com/KUChickCheck/liveness-model.git 
cd liveness-model
```

### 2️⃣ Create a Virtual Environment
Create a virtual environment in the project directory to isolate dependencies. Run the following commands:
- For Windows:
```bash
python -m venv venv
```
- For Mac/Linux:
```bash
python3 -m venv venv
```

### 3️⃣ Activate the Virtual Environment
Activate the virtual environment:
- For Windows:
```bash
.\venv\Scripts\activate
```
- For Mac/Linux:
```bash
source venv/bin/activate
```

Once activated, your terminal prompt should show the venv prefix, indicating that you're working inside the virtual environment.

### 4️⃣ Install dependencies
Install the required dependencies using pip:
```bash
pip install -r requirements.txt
```

### 5️⃣ Install Git LFS
If you haven't already, install Git LFS (Large File Storage) to manage large files like the model file (`.pth`).
- For Window: [Git LFS for Windows](https://git-lfs.github.com/)
- For Mac:
```bash
brew install git-lfs
```
- For Linux:
```bash
sudo apt-get install git-lfs
```

### 6️⃣ Initialize Git LFS
Once Git LFS is installed, initialize it in your repository:
```bash
git lfs install
```

### 7️⃣ Pull Large Files
If the repository contains large files tracked by Git LFS, run the following command to pull them to your local machine:
```bash
git lfs pull
```
This will download the model file (model.pth) and any other large files that are being managed by Git LFS.

### 8️⃣ Run the FastAPI server
To start the model API with FastAPI, run the following command:
```bash
uvicorn main:app --reload  # Starts the FastAPI server
```
The model will now be accessible via http://127.0.0.1:8000.

### 9️⃣ Model Files
- Model file: The trained model file (`vit_liveness_detection_modelV2.pth`) is stored in the repository and can be found at:
`vit_liveness_detection_modelV2.pth`
- Jupyter Notebook: A Jupyter notebook (`liveness-detection-train.ipynb`) is provided to train and test the model. This notebook demonstrates the steps for preprocessing, training, and evaluating the model.
You can find it at:
`liveness-detection-train.ipynb`

### 🤝 Contributing
Feel free to fork the repo, make your changes, and submit a pull request. All contributions are welcome!