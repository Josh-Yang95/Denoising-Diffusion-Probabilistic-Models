from accelerate import Accelerator
from accelerate.logging import get_logger
from trainer import *
from config import get_config
from Datasets import CELEB

logger = get_logger(__name__)
args = get_config()


def main(args):
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator = Accelerator(mixed_precision=args.mixed_precision, logging_dir=logging_dir)

    dataset = CELEB(size=128)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)

    trainer = Trainer(train_dataloader, args, accelerator)
    trainer.train()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank


if __name__ == "__main__":
    args = get_config()
    main(args)

# python main.py --train_data_dir celeb_data --train_batch_size 16 --output_dir="logs" --epochs=100 --gradient_accumulation_steps=1 --learning_rate=1e-4 --lr_warmup_steps=500 --mixed_precision=no
