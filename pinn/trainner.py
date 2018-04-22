import argparse, os, pinn, glob
from pinn.dataset import from_ani
"""
Usage:
gcloud ml-engine jobs submit training $JOB_NAME \
    --job-dir $JOB_DIR\
    --package-path $TRAINER_PACKAGE_PATH \
    --module-name $MAIN_TRAINER_MODULE \
    -- \
    --data data/*[123].h5 \
    --data_format ani \
    --n_epoch 10 \
    --max_steps 10000 \
    --log_interval 10 \
    --chk_interval 1000
"""
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir', type=str, dest='job_dir')
    parser.add_argument('--data', type=str, dest='data')
    parser.add_argument('--n_epoch', type=int, dest='n_epoch')
    parser.add_argument('--max_steps', type=int, dest='max_steps')
    parser.add_argument('--data_format', type=str, dest='data_format')
    parser.add_argument('--log_interval', type=int, dest='log_interval')
    parser.add_argument('--chk_interval', type=int, dest='chk_interval')
    args = parser.parse_args()
    files = sorted(glob.glob(args.data))
    files = ['gs://pinn-data/ANI-1_release/ani_gdb_s0%i.h5'%(i+1) for i in range(3)]
    #print(args.data)
    print(files)
    dataset = from_ani(files)
    model = pinn.pinn_model()
    model.train(dataset, log_dir=args.job_dir, max_steps=10000, n_epoch=100)
main()
