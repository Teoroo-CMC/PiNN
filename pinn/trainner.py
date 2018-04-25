import argparse, os, pinn, glob
from pinn.dataset import from_tfrecord_ani
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
    # Job
    parser.add_argument('--data-dir', type=str, dest='data_dir')
    parser.add_argument('--job-dir', type=str, dest='job_dir', default='.')
    parser.add_argument('--log-dir', type=str, dest='log_dir', default='logs')
    parser.add_argument('--chk-dir', type=str, dest='chk_dir', default='chks')
    parser.add_argument('--job-name', type=str, dest='job_name', default='training')
    parser.add_argument('--load-model', type=str, dest='load_model', default=None)
    # Algo
    parser.add_argument('--max_epoch', type=int, dest='max_epoch', default=10)
    parser.add_argument('--max_steps', type=int, dest='max_steps', default=1000)
    parser.add_argument('--log_interval', type=int, dest='log_interval', default=10)
    parser.add_argument('--chk_interval', type=int, dest='chk_interval', default=100)
    parser.add_argument('--learning_rate', type=float, dest='learning_rate', default=1e-4)
    parser.add_argument('--batch_size', type=int, dest='batch_size', default=1000)

    args = parser.parse_args()

    log_dir = os.path.join(args.job_dir, args.log_dir)
    chk_dir = os.path.join(args.job_dir, args.chk_dir)

    dataset = from_tfrecord_ani(args.data_dir)
    model = pinn.pinn_model()
    if args.load_model is not None:
        model.load(args.load_model)
    model.train(dataset,
                job_name=args.job_name,
                log_dir=log_dir,
                chk_dir=chk_dir,
                log_interval=args.log_interval,
                chk_interval=args.chk_interval,
                max_steps=args.max_steps,
                max_epoch=args.max_epoch,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size)
main()
