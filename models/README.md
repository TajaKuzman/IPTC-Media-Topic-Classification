# Models

The folder is dedicated to storing files with model parameters.

## Naming Convention

Each filename must follow the given naming convention:

```bash
# task   - the task the model was trained to solve
#          Possible tasks: NER, SSIM
# lmodel - The language model used to solve the task.
# tdata  - The short name of the dataset.
# ext    - The extension of the model file.
#          Possible options:
#            - pth: Pytorch extension.
#            - ckpt: Pytorch lightning extension.
#            - other: Any other extension.

{task}_{lmodel}_{tdata}.{ext}
```
