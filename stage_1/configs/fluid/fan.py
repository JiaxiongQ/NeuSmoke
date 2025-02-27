_base_ = './default.py'

expname = 'ours/fan'
basedir = './logs/fan2_ours2'

data = dict(
    datadir='/root/siton-gpfs-archive/qiujiaxiong/data_smoke/fan2',
    dataset_type='dnerf',
    white_bkgd=True,
)