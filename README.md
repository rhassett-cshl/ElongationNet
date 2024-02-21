# ElongationNet

## Quick Start
* Input dataset location:
  - /grid/siepel/home_norepl/hassett/ElongationNet/data/*.pkl
* Setup config json file in ./configs
  - Example below:
![image](https://github.com/rhassett-cshl/ElongationNet/assets/119357550/605f294c-28b1-41d5-b452-8fe4e846a6c6)

* Setup conda environment with environment.yml
* Train Model:
  - python main.py --mode=train --config_name=cnn_2

* Load model and save predictions to bigwig file:
  - python main.py --mode=save_results --config_name=cnn_2

 * Submit job to job scheduler
  - Example in submit_job.sh
 
* Model checkpoint saved to ./model_checkpoints folder after training with config_name.pth filename
* Bigwig files stored in ./results/config_name folder
