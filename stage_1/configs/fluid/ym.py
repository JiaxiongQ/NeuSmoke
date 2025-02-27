_base_ = './default.py'

expname = 'ours/double_vel'
basedir = '/root/siton-gpfs-archive/qiujiaxiong/data_smoke/s1n/yellow_mul1_ours_fean'

data = dict(
    datadir='/root/siton-gpfs-archive/qiujiaxiong/data_smoke/ym1',
    dataset_type='dnerf',
    white_bkgd=True,
)