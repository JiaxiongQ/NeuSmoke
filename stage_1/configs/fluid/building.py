_base_ = './default.py'

expname = 'ours/build'
basedir = './logs/build2_ours2'

data = dict(
    datadir='/root/siton-gpfs-archive/qiujiaxiong/data_smoke/building2',
    dataset_type='dnerf',
    white_bkgd=True,
)