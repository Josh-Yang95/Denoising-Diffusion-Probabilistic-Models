import os

import torch
import torch.nn.functional as F

from diffusions import UNet2DModel, DDIMScheduler, DDIMPipeline
from diffusions.optimization import get_scheduler
from diffusions.training_utils import EMAModel

from tqdm.auto import tqdm
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, data, args, accelerator):
        self.args = args
        self.accelerator = accelerator
        self.model = UNet2DModel(sample_size=args.resolution, in_channels=3, out_channels=3, layers_per_block=2,
                                 block_out_channels=(128, 128, 256, 256, 512, 512),
                                 down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D",
                                                   "AttnDownBlock2D", "DownBlock2D"),
                                 up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D",
                                                 "UpBlock2D", "UpBlock2D"))

        # self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000, tensor_format="pt")
        self.noise_scheduler = DDIMScheduler(num_train_timesteps=1000, tensor_format="pt")

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.learning_rate,
                                           betas=(args.adam_beta1, args.adam_beta2),
                                           weight_decay=args.adam_weight_decay,
                                           eps=args.adam_epsilon)

        self.lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=self.optimizer,
                                          num_warmup_steps=args.lr_warmup_steps,
                                          num_training_steps=(len(data) * args.num_epochs) // args.gradient_accumulation_steps)

        self.model, self.optimizer, self.dataloader, self.lr_scheduler = self.accelerator.prepare(self.model,
                                                                                                  self.optimizer, data,
                                                                                                  self.lr_scheduler)

        self.ema_model = EMAModel(self.model, inv_gamma=args.ema_inv_gamma, power=args.ema_power, max_value=args.ema_max_decay)

    def train(self):
        for epoch in range(self.args.num_epochs):
            self.model.train()
            self._run_epoch(self.dataloader, epoch)
            self.accelerator.wait_for_everyone()

            # Generate sample images for visual inspection
            if self.accelerator.is_main_process:
                if epoch % self.args.save_images_epochs == 0 or epoch == self.args.num_epochs - 1:
                    self.evaluation(epoch)
            self.accelerator.wait_for_everyone()

    def _run_epoch(self, data_loader, epoch):
        progress_bar = tqdm(total=len(data_loader), disable=not self.accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(data_loader):
            clean_images = batch['image'].permute(0, 3, 1, 2)  # batch[0]
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bsz = clean_images.shape[0]
            timesteps = torch.randint(
                0, self.noise_scheduler.num_train_timesteps, (bsz,), device=clean_images.device
            ).long()

            noisy_images = self.noise_scheduler.add_noise(clean_images, noise, timesteps)

            with self.accelerator.accumulate(self.model):
                noise_pred = self.model(noisy_images, timesteps)["sample"]
                loss = F.mse_loss(noise_pred, noise)
                self.accelerator.backward(loss)

                self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.lr_scheduler.step()
                if self.args.use_ema:
                    self.ema_model.step(self.model)
                self.optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": self.lr_scheduler.get_last_lr()[0]}
            if self.args.use_ema:
                logs["ema_decay"] = self.ema_model.decay
            progress_bar.set_postfix(**logs)
        progress_bar.close()

    def evaluation(self, epoch):
        pipeline = DDIMPipeline(
            unet=self.accelerator.unwrap_model(self.ema_model.averaged_model if self.args.use_ema else self.model),
            scheduler=self.noise_scheduler,
        )

        generator = torch.manual_seed(0)

        # run pipeline in inference (sample random noise and denoise)
        images = pipeline(generator=generator, batch_size=self.args.eval_batch_size, output_type="numpy")["sample"]
        images_processed = (images * 255).round().astype("uint8")

        save_dir = f'{self.args.output_dir}/{self.args.save_image_dir}/epoch-{epoch}'
        os.makedirs(save_dir, exist_ok=True)
        for idx, image in enumerate(images_processed):
            plt.imsave(f'{save_dir}/{idx}.jpg', image)

        if epoch % self.args.save_model_epochs == 0 or epoch == self.args.num_epochs - 1:
            pipeline.save_pretrained(self.args.output_dir)
