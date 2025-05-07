import os
import json
import tempfile
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

from models.mask_rcnn import CustomMaskRCNN
from utils.dataloader import (
    get_transforms,
    collate_fn,
    SegmentationDataset,
    encode_mask
)

# Define color mapping for visualization
COLORS = [
    (0, 0, 0),
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40)
]
CLASS_COLORS = ListedColormap([(r/255, g/255, b/255) for r, g, b in COLORS])


class Trainer:
    """
    Instance Segmentation Model Trainer

    Class for training and evaluating Mask R-CNN models, supports mixed
    precision training, validation, and testing.
    """
    def __init__(self, args):
        """
        Initialize trainer

        Args:
            args: Object containing all configuration parameters
        """
        self.args = args
        self.device = args.device
        self.best_map = 0.0

        # Initialize model
        self.model = CustomMaskRCNN(
            num_classes=args.num_classes,
            box_score_thresh=args.box_score_thresh,
            backbone=args.backbone,
            mask_head_type=args.mask_head_type,
            use_transformer=args.use_transformer,
            transformer_layers=args.transformer_layers,
            transformer_dim=args.transformer_dim,
            transformer_heads=args.transformer_heads,
            use_focal_loss=(
                args.use_focal_loss if hasattr(args, 'use_focal_loss')
                else False
            ),
            focal_loss_alpha=(
                args.focal_loss_alpha if hasattr(args, 'focal_loss_alpha')
                else 0.25
            ),
            focal_loss_gamma=(
                args.focal_loss_gamma if hasattr(args, 'focal_loss_gamma')
                else 2.0
            ),
        ).to(self.device)

        # Get data transformation functions
        self.train_transform, self.test_transform = get_transforms()

        # Set up training or testing environment based on mode
        if not args.eval:
            self._setup_training()
        else:
            self._setup_testing()

    def _setup_training(self):
        """
        Set up training environment, including optimizer, scheduler,
        datasets, and logging
        """
        self.optimizer = self._create_optimizer()
        self._setup_scheduler()
        self._prepare_datasets()
        self._setup_dataloaders()
        self._setup_logging()

    def _setup_testing(self):
        """Set up testing environment, only load test dataset"""
        self.test_dataset = SegmentationDataset(
            root_dir=self.args.test_data_dir,
            transform=self.test_transform,
            is_test=True,
            test_json=self.args.json_map_imgname_to_id,
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=16,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def _create_optimizer(self):
        """
        Create optimizer

        Returns:
            torch.optim.Optimizer: AdamW optimizer
        """
        if (self.args.differential_lr and
                hasattr(self.model, 'get_parameter_groups')):
            # Use different learning rates for backbone and other parts
            param_groups = self.model.get_parameter_groups(
                backbone_lr_factor=self.args.backbone_lr_factor
            )
            optimizer = torch.optim.AdamW(
                [
                    {
                        'params': group['params'],
                        'lr': self.args.lr * group['lr_factor']
                    }
                    for group in param_groups
                ],
                weight_decay=self.args.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )

            print("Using differential learning rates:")
            for i, group in enumerate(param_groups):
                print(
                    f"  Group {i+1}: {len(group['params'])} parameters, "
                    f"lr_factor={group['lr_factor']}"
                )
        else:
            # Basic version: all parameters use the same learning rate
            params = [p for p in self.model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(
                params,
                lr=self.args.lr,
                weight_decay=self.args.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )

        # Apply layer freezing if specified
        if (hasattr(self.args, 'freeze_backbone_layers') and
                self.args.freeze_backbone_layers > 0):
            self.model.freeze_backbone(layers=self.args.freeze_backbone_layers)

        return optimizer

    def _setup_scheduler(self):
        """Set up learning rate scheduler"""
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.args.epochs,
            eta_min=1e-6,
        )

    def _prepare_datasets(self):
        """Prepare training and validation datasets"""
        # Load complete dataset
        dataset = SegmentationDataset(root_dir=self.args.train_data_dir)

        # Split dataset into training and validation sets
        val_size = int(len(dataset) * 0.1)  # Use 10% of data as validation set
        train_size = len(dataset) - val_size
        self.train_dataset, self.valid_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

        # Apply different data transformations
        self.train_dataset.dataset.transform = self.train_transform
        self.valid_dataset.dataset.transform = self.test_transform

    def _setup_dataloaders(self):
        """Create training and validation data loaders"""
        # Training data loader
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=16,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        # Validation data loader
        self.val_loader = DataLoader(
            self.valid_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=16,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def _setup_logging(self):
        """Set up TensorBoard logging and checkpoint saving directory"""
        # Create output directory
        os.makedirs(self.args.output_dir, exist_ok=True)

        # Create visualization directories
        os.makedirs(os.path.join(self.args.output_dir, "viz"), exist_ok=True)
        os.makedirs(os.path.join(self.args.output_dir, "plots"), exist_ok=True)

        # Initialize metrics tracking
        self.epoch_losses = []
        self.epoch_maps = []
        self.epoch_nums = []

        # Set up TensorBoard logging
        self.writer = SummaryWriter(log_dir=self.args.output_dir)

    def train(self, val_every_epochs=1):
        """
        Train model

        Args:
            val_every_epochs (int): Validate every this many epochs
        """
        # Initialize mixed precision training
        scaler = GradScaler()

        # Training loop
        for epoch in range(self.args.epochs):
            # Train one epoch
            train_metrics = self._train_epoch(epoch, scaler)

            # Validate if needed
            if (epoch + 1) % val_every_epochs == 0:
                val_map = self._validate_epoch(epoch)
                self._log_metrics(epoch, train_metrics, val_map)

                # Update best model if needed
                self._update_best_model(val_map, epoch)

                # Visualize some examples
                self._visualize_segmentation(
                    next(iter(self.val_loader)),
                    epoch
                )
            else:
                self._log_metrics(epoch, train_metrics, None)

            # Update learning rate
            self.scheduler.step()

        # Save final training plots
        self._save_training_plots()

    def _train_epoch(self, epoch, scaler):
        """
        Train for one epoch

        Args:
            epoch (int): Current epoch number
            scaler (GradScaler): Mixed precision scaler

        Returns:
            dict: Training metrics
        """
        self.model.train()
        metrics = {
            'loss': 0.0,
            'loss_classifier': 0.0,
            'loss_box_reg': 0.0,
            'loss_mask': 0.0,
            'loss_objectness': 0.0,
            'loss_rpn_box_reg': 0.0
        }

        # Training loop
        for images, targets in tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{self.args.epochs}"
        ):
            # Move data to device
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()}
                       for t in targets]

            # Compute loss with mixed precision
            loss_dict = self._compute_loss(images, targets, scaler)

            # Update metrics
            self._update_metrics(metrics, loss_dict)

        # Average metrics
        num_batches = len(self.train_loader)
        return {k: v / num_batches for k, v in metrics.items()}

    def _compute_loss(self, images, targets, scaler):
        """
        Compute loss with mixed precision

        Args:
            images (list): List of images
            targets (list): List of target dictionaries
            scaler (GradScaler): Mixed precision scaler

        Returns:
            dict: Loss dictionary
        """
        # Forward pass with mixed precision
        with autocast():
            loss_dict = self.model(images, targets)
            loss = sum(loss_dict.values())

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(self.optimizer)
        scaler.update()
        self.optimizer.zero_grad()

        return loss_dict

    def _update_metrics(self, metrics, loss_dict):
        """
        Update metrics dictionary

        Args:
            metrics (dict): Current metrics
            loss_dict (dict): Loss dictionary from model
        """
        for k, v in loss_dict.items():
            metrics[k] += v.item()

    def _validate_epoch(self, epoch):
        """
        Validate model for one epoch

        Args:
            epoch (int): Current epoch number

        Returns:
            float: Mean Average Precision (mAP)
        """
        self.model.eval()

        # Create COCO ground truth
        coco_gt = self._create_coco_gt()

        # Initialize predictions list
        preds = []

        # Validation loop
        with torch.no_grad():
            for images, targets, image_ids in tqdm(
                self.val_loader,
                desc=f"Validating epoch {epoch + 1}"
            ):
                # Move data to device
                images = [img.to(self.device) for img in images]

                # Get predictions
                predictions = self.model(images)

                # Process predictions
                self._process_predictions(
                    predictions,
                    targets,
                    image_ids,
                    preds,
                    coco_gt
                )

        # Compute mAP
        return self._compute_map(preds, coco_gt)

    def _create_coco_gt(self):
        """
        Create COCO format ground truth

        Returns:
            dict: COCO format ground truth dictionary
        """
        return {
            "images": [],
            "annotations": [],
            "categories": [
                {"id": i, "name": f"class_{i}"}
                for i in range(1, 5)
            ],
        }

    def _process_predictions(
        self,
        predictions,
        targets,
        image_ids,
        preds,
        coco_gt
    ):
        """
        Process predictions, add to prediction list and ground truth

        Args:
            predictions: Model predictions
            targets: Ground truth
            image_ids: List of image IDs
            preds: Prediction result list
            coco_gt: COCO format ground truth dictionary
        """
        # Start from current annotation count for new annotation IDs
        ann_id = len(coco_gt["annotations"]) + 1

        # Process each image's predictions and ground truth
        for i, (image_id, pred) in enumerate(zip(image_ids, predictions)):
            self._add_gt_annotations(targets[i], image_id, ann_id, coco_gt)
            self._add_predictions(pred, image_id, preds)

    def _add_gt_annotations(self, target, image_id, ann_id, coco_gt):
        """
        Add ground truth to COCO GT

        Args:
            target: Target dictionary
            image_id: Image ID
            ann_id: Starting annotation ID
            coco_gt: COCO format ground truth dictionary
        """
        # Add image information
        coco_gt["images"].append({
            "id": image_id,
            "width": target["masks"].shape[2],
            "height": target["masks"].shape[1],
        })

        # Get ground truth masks and labels
        gt_masks = target["masks"].numpy()
        gt_labels = target["labels"].numpy()

        # Add annotations for each instance
        for mask, label in zip(gt_masks, gt_labels):
            encoded_mask = encode_mask(mask)
            coco_gt["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "bbox": list(maskUtils.toBbox(encoded_mask)),
                "category_id": int(label),
                "segmentation": encoded_mask,
                "iscrowd": 0,
                "area": int(mask.sum()),
            })
            ann_id += 1

    def _add_predictions(self, pred, image_id, preds):
        """
        Add predictions to prediction list

        Args:
            pred: Model predictions
            image_id: Image ID
            preds: Prediction result list
        """
        # Process each predicted instance
        for j in range(len(pred["masks"])):
            mask = pred["masks"][j].cpu().numpy()[0]
            binary_mask = (mask >= 0.5).astype(np.uint8)
            encoded_mask = encode_mask(binary_mask)
            bbox = maskUtils.toBbox(encoded_mask)

            preds.append({
                "image_id": image_id,
                "score": pred["scores"][j].item(),
                "category_id": pred["labels"][j].item(),
                "segmentation": encoded_mask,
                "bbox": bbox.tolist()  # Add bbox information
            })

    def _compute_map(self, preds, coco_gt):
        """
        Calculate mAP

        Args:
            preds: Prediction result list
            coco_gt: COCO format ground truth dictionary

        Returns:
            float: mAP value
        """
        # If there are no prediction results, return 0
        if not preds:
            return 0.0

        # Use temporary file to calculate mAP
        with tempfile.NamedTemporaryFile(
            mode="w+",
            suffix=".json"
        ) as pred_f, tempfile.NamedTemporaryFile(
            mode="w+",
            suffix=".json"
        ) as gt_f:
            json.dump(preds, pred_f)
            json.dump(coco_gt, gt_f)

            pred_f.flush()
            gt_f.flush()

            # Redirect stdout to capture COCO output
            import sys
            import io
            original_stdout = sys.stdout
            sys.stdout = io.StringIO()

            try:
                # Use COCO evaluation tool to calculate mAP
                coco_gt = COCO(gt_f.name)
                coco_dt = coco_gt.loadRes(pred_f.name)
                coco_eval = COCOeval(coco_gt, coco_dt, iouType="segm")

                # Use standard COCO evaluation parameters
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()

                # Print detailed evaluation metrics
                print("\nDetailed COCO Evaluation Metrics:")
                print(
                    f"Average Precision (AP) @ IoU=0.50: "
                    f"{coco_eval.stats[0]:.4f}"
                )
                print(
                    f"Average Precision (AP) @ IoU=0.75: "
                    f"{coco_eval.stats[1]:.4f}"
                )
                print(
                    f"Average Precision (AP) @ IoU=0.50:0.95: "
                    f"{coco_eval.stats[2]:.4f}"
                )
                print(
                    f"Average Recall (AR) @ IoU=0.50:0.95: "
                    f"{coco_eval.stats[8]:.4f}"
                )

                map_value = coco_eval.stats[0]
            finally:
                # Restore stdout
                sys.stdout = original_stdout

            return map_value

    def _update_best_model(self, val_map, epoch):
        """
        If current mAP is better, update best model

        Args:
            val_map (float): Validation set mAP
            epoch (int): Current epoch
        """
        if val_map > self.best_map:
            self.best_map = val_map
            self.save_weights()

    def _log_metrics(self, epoch, train_metrics, val_map):
        """
        Record training metrics

        Args:
            epoch (int): Current epoch
            train_metrics (dict): Training metrics dictionary
            val_map (float): Validation set mAP
        """
        # Simple output format
        lr = self.optimizer.param_groups[0]["lr"]
        total_loss = train_metrics["loss"]
        print(
            f"Epoch[{epoch:>2}/{self.args.epochs:<2}] "
            f"Loss: {total_loss:.4f} | "
            f"mAP: {val_map:.4f} | "
            f"Best: {self.best_map:.4f} | "
            f"LR: {lr:.6f}"
        )

        # Record metric values for plotting
        self.epoch_losses.append(total_loss)
        self.epoch_maps.append(val_map)
        self.epoch_nums.append(epoch)

        # Record all details to TensorBoard
        for name, value in train_metrics.items():
            self.writer.add_scalar(f"Loss/{name}", value, epoch)
        self.writer.add_scalar("Learning_Rate", lr, epoch)
        self.writer.add_scalar("mAP/val", val_map, epoch)

        # Save plots every 5 epochs and on last epoch
        if epoch % 5 == 0 or epoch == self.args.epochs:
            self._save_training_plots()

    def _visualize_segmentation(self, example, epoch):
        """
        Visualize segmentation results for validation example

        Args:
            example: Validation example containing image, target, and
                prediction
            epoch (int): Current epoch
        """
        print("Visualizing segmentation results...")

        # Get image, target masks, and prediction masks
        img = example['image'].permute(1, 2, 0).numpy()
        # Denormalize image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1)

        # Get ground truth masks and labels
        gt_masks = example['target']['masks'].numpy()
        gt_labels = example['target']['labels'].numpy()

        # Get prediction masks and labels
        pred_masks = example['prediction']['masks'].cpu().numpy()
        pred_labels = example['prediction']['labels'].cpu().numpy()
        pred_scores = example['prediction']['scores'].cpu().numpy()

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot original image
        axes[0].imshow(img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Plot ground truth masks
        gt_overlay = np.zeros_like(img)
        for mask, label in zip(gt_masks, gt_labels):
            color = CLASS_COLORS(label)
            gt_overlay[mask > 0.5] = color
        axes[1].imshow(img)
        axes[1].imshow(gt_overlay, alpha=0.5)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')

        # Plot prediction masks
        pred_overlay = np.zeros_like(img)
        for mask, label, score in zip(pred_masks, pred_labels, pred_scores):
            if score > 0.5:  # Only show high confidence predictions
                color = CLASS_COLORS(label)
                pred_overlay[mask > 0.5] = color
        axes[2].imshow(img)
        axes[2].imshow(pred_overlay, alpha=0.5)
        axes[2].set_title('Prediction')
        axes[2].axis('off')

        # Save figure
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.args.output_dir, 'viz', f'epoch_{epoch}.png')
        )
        plt.close()

    def _save_training_plots(self):
        """Save training plots"""
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))

        # Plot loss curves
        axes[0].plot(self.epoch_nums, self.epoch_losses, 'b-', label='Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].grid(True)
        axes[0].legend()

        # Plot mAP curves
        axes[1].plot(self.epoch_nums, self.epoch_maps, 'r-', label='mAP')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('mAP')
        axes[1].set_title('Validation mAP')
        axes[1].grid(True)
        axes[1].legend()

        # Save figure
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.args.output_dir, 'plots', 'training_curves.png')
        )
        plt.close()

    @torch.no_grad()
    def test(self, load_weights=True):
        """
        Test model on test dataset

        Args:
            load_weights (bool): Whether to load best weights before testing
        """
        if load_weights:
            self.load_weights()

        self.model.eval()
        preds = []

        # Test loop
        for images, image_ids in tqdm(
            self.test_loader,
            desc="Testing"
        ):
            # Move data to device
            images = [img.to(self.device) for img in images]

            # Get predictions
            predictions = self.model(images)

            # Process predictions
            for pred, image_id in zip(predictions, image_ids):
                self._add_predictions(pred, image_id, preds)

        # Save results
        self._save_test_results(preds)

    def _save_test_results(self, preds):
        """
        Save test results to JSON file

        Args:
            preds (list): List of predictions
        """
        # Save predictions
        with open(
            os.path.join(self.args.output_dir, 'test_results.json'),
            'w'
        ) as f:
            json.dump(preds, f)

    def save_weights(self):
        """Save model weights"""
        # Create checkpoint directory
        os.makedirs(
            os.path.join(self.args.output_dir, 'checkpoints'),
            exist_ok=True
        )

        # Save checkpoint
        torch.save(
            {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'epoch': self.args.epochs,
                'best_map': self.best_map,
            },
            os.path.join(
                self.args.output_dir,
                'checkpoints',
                'model.pth'
            )
        )

    def load_weights(self):
        """Load model weights"""
        # Load checkpoint
        checkpoint = torch.load(
            os.path.join(
                self.args.output_dir,
                'checkpoints',
                'model.pth'
            )
        )

        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load scheduler state
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Load best mAP
        self.best_map = checkpoint['best_map']
