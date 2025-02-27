_base_ = './default2.py'

expname = 'ours/sphere_vel_fean'
basedir = '/root/siton-gpfs-archive/qiujiaxiong/data_smoke/s1n/logs/scalar_pi'

data = dict(
    datadir='/root/siton-gpfs-archive/qiujiaxiong/data_smoke/pinf_data/data1',
    dataset_type='dnerf',
    white_bkgd=False,
)