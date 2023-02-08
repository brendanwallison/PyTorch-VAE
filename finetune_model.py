from experiment import VAEXperiment
import yaml
import torch 
from models import *
from dataset import *
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from dataset import VAEDataset, VAEDataset1D
from collections import OrderedDict


if __name__ == '__main__':
    config = yaml.safe_load(open('./configs/vae_1d.yaml'))
    model = vae_models[config['model_params']['name']](**config['model_params'])
    experiment = VAEXperiment(model, config['exp_params'])
    if 'custom_params' in config:
        if config['custom_params']['resume_training']:
            ckpt = torch.load(config['custom_params']['resume_chkpt_path'])
            experiment.load_state_dict(ckpt['state_dict'])
    data = VAEDataset1D(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0)
    data.setup()
    tb_logger =  TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                                name=config['model_params']['name'],)
    runner = Trainer(logger=tb_logger,
                    callbacks=[
                        LearningRateMonitor(),
                        ModelCheckpoint(save_top_k=2, 
                                        dirpath =os.path.join(tb_logger.log_dir , "checkpoints"), 
                                        monitor= "val_loss",
                                        save_last= True),
                    ],
                    #strategy=DDPStrategy(find_unused_parameters=False),
                    **config['trainer_params'])

    Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)

    print(f"======= Training {config['model_params']['name']} =======")
    runner.fit(experiment, datamodule=data)