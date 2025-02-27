_base_ = './default.py'

expname = 'ours/car'
basedir = '/root/siton-gpfs-archive/qiujiaxiong/data_smoke/s1n/car'

data = dict(
    datadir='/root/siton-gpfs-archive/qiujiaxiong/data_smoke/car',
    dataset_type='dnerf',
    white_bkgd=True,
)