# Embryo stage classification
Model for classifying Drosophila embryo stages

## Usage:

* install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) on your laptop, if you don't have it yet
* install a new environment from the embryo_class.yml as follows:

`conda env create -f embryo_class.yml`

* activate the environment:

`conda activate embryo_class`

* use the script as follows:

`python predict_from_folder.py /path/to/folder_with_images`

## Script options:

* --device - cpu by default, can be set to GPU number (requires a different conda env)
* --batch_size - 10 by default, reduce if script crashes/stalls
* --num_workers - 10 by default, reduce if script crashes/stalls
* --model - stage_classifier.pth by default, change when using a different model or if the model is in a different directory
