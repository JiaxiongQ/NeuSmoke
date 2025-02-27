_base_ = './default.py'

expname = 'ours/double'
basedir = './logs/double_ours'

data = dict(
    datadir='/root/siton-gpfs-archive/qiujiaxiong/data_smoke/double2c',
    dataset_type='dnerf',
    white_bkgd=True,
)