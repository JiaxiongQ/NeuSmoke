_base_ = './default.py'

expname = 'ours/double_vel'
basedir = '/root/siton-gpfs-archive/qiujiaxiong/data_smoke/s1n/double_abs_tc5'

data = dict(
    datadir='/root/siton-gpfs-archive/qiujiaxiong/data_smoke/double_multi1',
    dataset_type='dnerf',
    white_bkgd=True,
)