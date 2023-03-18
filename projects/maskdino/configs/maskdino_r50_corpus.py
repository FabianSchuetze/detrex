from detrex.config import get_config
from .models.maskdino_r50 import model
from .data.corpus_instance_seg import dataloader
# from .data.coco_instance_seg import dataloader

from fvcore.common.param_scheduler import MultiStepParamScheduler
from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler

dataloader = get_config("common/data/custom.py").dataloader
train = get_config("common/train.py").train
# max training iterations
train.max_iter = 368750
# warmup lr scheduler
lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1],
        milestones=[327778, 355092],
    ),
    warmup_length=10 / train.max_iter,
    warmup_factor=1.0,
)

optimizer = get_config("common/optim.py").AdamW
# lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_50ep

# initialize checkpoint to be loaded
train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
train.output_dir = "./output/maskdino_r50_corpus"


# run evaluation every 5000 iters
train.eval_period = 2000

# log training infomation every 20 iters
train.log_period = 50

# save checkpoint every 5000 iters
train.checkpointer.period = 2000

# gradient clipping for training
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.01
train.clip_grad.params.norm_type = 2

# set training devices
train.device = "cuda"

# # please notice that this is total batch size.
# # surpose you're using 4 gpus for training and the batch size for
# # each gpu is 16/4 = 4
dataloader.train.total_batch_size = 8

# modify optimizer config
optimizer.lr = 1e-4 * dataloader.train.total_batch_size / 16
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

# # modify dataloader config
dataloader.train.num_workers = 8
#

# dump the testing results into output_dir for visualization
dataloader.evaluator.output_dir = train.output_dir
