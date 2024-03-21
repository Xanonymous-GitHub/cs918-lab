# Assignment 2

### Get started

1. Install dependencies.
   
  This behavior is included in the notebook, but you can also run it manually.
  ```bash
  # If you use any kind of virtual environment, activate it first.
  pip install -r requirements.txt --no-cache-dir
  ```

2. (Optional) Download trained models.

  If you want to skip training and use the provided models, download them from https://huggingface.co/Xanonymous/cs918-assignment2
  and place them in the same folder as the notebook.

3. Provide test data file(s)

  Create a folder named `data` in the same directory as the notebook, and place the dataset in this location.
  Please note, you do not need to provide the GloVe data, as the notebook will download these files on its own. We use the data provided [here](https://huggingface.co/stanfordnlp/glove/blob/main/glove.6B.zip).
  In the notebook's `Define Global instances` cell, append the new file names to the `test_set_names` list.
  The code will automatically load the data and run, you don't need to change anything else.

4. Run the notebook/lab server.

5. (Optional) select which models should be trained.

  In the notebook's `Define Global instances` cell, change flags to `False` for the models you don't want to train. Otherwise, all models will be trained.
  If skipping training, make sure the trained model files are available in the same folder as the notebook.

### Note
- The notebook has defined a dataset folder in `Define Global instances` cell. The default dataset folder is `./data/`. If you want to use another folder, change the `dataset_base_path` variable.
- The given skeleton code has been modified or removed to fit the requirements of the assignment. But the notebook will still show the F1 Macro score and the confusion matrix for each model.

### Environment info
The following information is used for developing this notebook. You may not need to follow these versions strictly, but it is recommended to use the same versions to avoid any compatibility issues.

```yaml
Kernel: Linux dcs-panda-01 4.18.0-513.18.1.el8_9.x86_64
CPU: Intel(R) Core(TM) i5-8500 CPU @ 3.00GHz x86_64 6 cores
GPU: NVIDIA GeForce GTX 1080 8192MiB
CUDA: YES, version 12.2
Python: Python 3.12.2 (main, Mar  8 2024, 12:23:34) [GCC 12.2.0]
```
