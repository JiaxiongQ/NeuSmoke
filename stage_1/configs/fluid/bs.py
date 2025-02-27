_base_ = './default.py'

expname = 'ours/bs_vel'
basedir = './logs/bs_our'

data = dict(
    datadir='/home/ubuntu/Documents/fluid/data_our/blue_smoke',
    dataset_type='dnerf',
    white_bkgd=True,
)