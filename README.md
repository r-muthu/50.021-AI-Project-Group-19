# 🧠 Hate Speech Detection Project by Group 19

1007010 Muthu Ramaswamy \
1007124 Manishansh Shaswat \
1007012 Mohamed Zuhairi Bin Mohamed Amran \
1007156 Alen Syzdykov \
1006324 Ethan Choo E-Rhen 

---

## 🛡️ Our best model: distilRoBERTa

This project enables users to train, test, and deploy a robust hate speech classification model using `distilroberta-base`. It leverages Hugging Face's Transformers and PyTorch, with gradual unfreezing, early stopping, and safetensors support for efficient model saving.

---

## 📁 Project Structure

- `modelSetup_Train.ipynb`: Notebook to train, save, and test models.
- `models.py`: Contains model architectures.
- `data_preprocessing.py`: Dataset handling and preprocessing logic.
- `Hate_Speech_UI.py`: Streamlit web app for deploying a trained model.
- `HateSpeechDatasetBalanced.csv`: The dataset file used for training and evaluation.

---

## ⚙️ Prerequisites

Install the required Python libraries:

```bash
pip install -U transformers datasets evaluate accelerate scikit-learn matplotlib safetensors
```

---

## 🧠 Using the Pretrained Model (Recommended)

If you'd like to skip training and immediately test the model, you can download the pretrained weights:

1. **Download weights (.safetensors)**:
   [Download model.safetensors](https://drive.google.com/file/d/1c7pxEXCaEclE-OtdFrGTInDKUWgEbViM/view?usp=sharing)

2. **Place the file** in your project directory under `models/`:
   ```
   ./models/model.safetensors
   ```

3. **Edit `Hate_Speech_UI.py` to load the weights**:
   ```python
   state_dict = load_file("models/model.safetensors")
   model.load_state_dict(state_dict, strict=False)
   ```

4. **Run the Streamlit app**:
   ```bash
   streamlit run Hate_Speech_UI.py
   ```

---

## 🏋️‍♂️ Option B: Train the Model from Scratch

1. Open `modelSetup_Train.ipynb`.

2. Follow the **section flow** as outlined in the notebook:
   - Skip the **"Loading of model"** section.
   - Run all other sections **sequentially**.

3. Ensure this line is set for the best-performing model:
   ```python
   model_checkpoint = "distilroberta-base"
   ```
   
   Do not run this cell:
   ```python
   # To initialise a model with individual transformer layers, run this cell (3 layers)
   model = SimpleTransformerModel(vocab_size, num_labels=2, dropout=0.1, num_layers=3)
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model = model.to(device)
   ```

5. The training will:
   - Use a max sequence length of 512 tokens.
   - Use a batch size of 16 and train for up to 20 epochs.
   - Apply early stopping with patience of 2.
   - Save the best model using `.safetensors`.

6. After training completes, save the model using:
   ```python
   train_log.save_model("./models/distilroberta-base-safetensors")
   ```

---

## 💾 Saving Model Weights

In `modelSetup:Train.ipynb`, after training:

1. Convert the model to `.safetensors` (optional, already done during training if using safetensors).
2. Save the state dict:
   ```python
   from safetensors.torch import save_file
   state_dict = model.state_dict()
   save_file(state_dict, "./models/model.safetensors")
   ```

---

## 🔁 Loading and Testing the Model

To test the saved model:

1. Set the correct path:
   ```python
   weights_location = "./models/model.safetensors"
   ```

2. Load the model:
   ```python
   from safetensors.torch import load_file
   state_dict = load_file(weights_location)
   model.load_state_dict(state_dict, strict=False)
   ```

3. Run the final cells in `train.ipynb` to evaluate test performance.

---

## 🌐 Deploying the Model via Streamlit

1. Open `Hate_Speech_UI.py`.

2. Update the path to your saved `.safetensors` file:
   ```python
   state_dict = load_file("models/model.safetensors")
   ```

3. Run the app:
   ```bash
   streamlit run Hate_Speech_UI.py
   ```

4. Use the UI to input text and view real-time predictions.

---

## 🧪 Model Info

- **Model**: `distilroberta-base`
- **Max Length**: 512
- **Epochs**: Up to 20 with early stopping
- **Batch Size**: 16
- **Tokenizer Resize**: Automatically handled
- **Callbacks**:
  - Gradual unfreezing (optional toggle)
  - Early stopping
- **Save Format**: `.safetensors`

---

## 📊 Evaluation

Evaluation includes:
- Accuracy and F1-score on test set

These are automatically generated in the final cells of `modelSetup_Train.ipynb`.
