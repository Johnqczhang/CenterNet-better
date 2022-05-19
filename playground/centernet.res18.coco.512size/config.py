import os.path as osp
from dl_lib.configs.base_detection_config import BaseDetectionConfig

_config_dict = dict(
    MODEL=dict(
        WEIGHTS="",
        RESNETS=dict(DEPTH=18),
        PIXEL_MEAN=[123.675, 116.28, 103.53],
        PIXEL_STD=[58.395, 57.12, 57.375],
        CENTERNET=dict(
            DECONV_CHANNEL=[512, 256, 128, 64],
            DECONV_KERNEL=[4, 4, 4],
            NUM_CLASSES=80,
            MODULATE_DEFORM=True,
            BIAS_VALUE=-2.19,
            DOWN_SCALE=4,
            MIN_OVERLAP=0.7,
            TENSOR_DIM=128,
        ),
        LOSS=dict(
            CLS_WEIGHT=1,
            WH_WEIGHT=0.1,
            REG_WEIGHT=1,
        ),
    ),
    INPUT=dict(
        AUG=dict(
            TRAIN_PIPELINES=[
                ('CenterAffine', dict(
                    boarder=128,
                    output_size=(512, 512),
                    random_aug=True)),
                ('RandomFlip', dict()),
                ('RandomBrightness', dict(intensity_min=0.6, intensity_max=1.4)),
                ('RandomContrast', dict(intensity_min=0.6, intensity_max=1.4)),
                ('RandomSaturation', dict(intensity_min=0.6, intensity_max=1.4)),
                ('RandomLighting', dict(scale=0.1)),
            ],
            TEST_PIPELINES=[
            ],
        ),
        FORMAT="RGB",
        OUTPUT_SIZE=(128, 128),
    ),
    DATALOADER=dict(
        NUM_WORKERS=4,
    ),
    DATASETS=dict(
        TRAIN=("coco_2017_train",),
        TEST=("coco_2017_val",),
    ),
    SOLVER=dict(
        OPTIMIZER=dict(
            NAME="SGD",
            BASE_LR=0.02,
            WEIGHT_DECAY=1e-4,
        ),
        LR_SCHEDULER=dict(
            GAMMA=0.1,
            # STEPS=(81000, 108000),
            # MAX_ITER=126000,
            STEPS=(82440, 109920),
            MAX_ITER=128240,  # 140 epochs
            WARMUP_ITERS=1000,
        ),
        IMS_PER_BATCH=128,
        CHECKPOINT_PERIOD=4580,
    ),
    OUTPUT_DIR=osp.join(
        '/data/Outputs/model_logs/playground',
        osp.split(osp.realpath(__file__))[0].split("playground/")[-1]
    ),
    GLOBAL=dict(DUMP_TEST=False),
    TEST=dict(
        EVAL_PERIOD=4580,
    )
)


class CenterNetConfig(BaseDetectionConfig):
    def __init__(self):
        super(CenterNetConfig, self).__init__()
        self._register_configuration(_config_dict)


config = CenterNetConfig()
