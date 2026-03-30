# EmbeddedMachineLearning
Embedded ML with jupyternotebook and TFLite and STM32CubeIDE.AI

# How to Run
In Jupyter Notebook
Open Anaconda Prompt.

Launch Jupyter Notebook.

Open lab4.ipynb.

Run the notebook cells in order.

# Save Validation Data
The notebook will generate:

Data/images/

Data/models/own_cifar10_model.h5

Data/own_cifar10_validation_20image.csv

Data/labels/own_cifar10_labels.txt

# Deploy the Model
Open STM32CubeMX.

Select the B-L475E-IOT01A2 board.

Enable CubeAI under software packs.

Load the .h5 model.

Analyze and generate code.

Open the generated project in STM32CubeIDE.

Flash the board.

# Test Inference
Install the required Python packages:

bash
python -m pip install -U opencv-python protobuf tqdm==4.50.2
Go to the Misc folder.

# Run the AI runner:

bash
python ui_python_ai_runner.py
Refresh NN and camera on the board tool.

Load the model and label file.

Open an image or use the camera to test classification.
