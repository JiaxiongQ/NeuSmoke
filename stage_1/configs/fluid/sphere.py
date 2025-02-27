_base_ = './default.py'

expname = 'ours/sphere_vel_fean8'
basedir = './logs/sphere_fean'

data = dict(
    datadir='/root/siton-gpfs-archive/qiujiaxiong/data_smoke/pinf_data/data2/sphere',
    dataset_type='dnerf',
    white_bkgd=True,
)