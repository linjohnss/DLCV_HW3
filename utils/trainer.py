import os
import time
import json
import tempfile
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
from tqdm import tqdm
from torch import nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
import cv2
from PIL import Image

from models.mask_rcnn import CustomMaskRCNN
from utils.dataloader import get_transforms, collate_fn, SegmentationDataset, encode_mask

# 定義用於可視化的顏色映射
COLORS = [(0, 0, 0), (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40)]
CLASS_COLORS = ListedColormap([(r/255, g/255, b/255) for r, g, b in COLORS])

class Trainer:
    """
    Instance Segmentation Model Trainer
    
    Class for training and evaluating Mask R-CNN models, supports mixed precision training, validation, and testing.
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
            use_focal_loss=args.use_focal_loss if hasattr(args, 'use_focal_loss') else False,
            focal_loss_alpha=args.focal_loss_alpha if hasattr(args, 'focal_loss_alpha') else 0.25,
            focal_loss_gamma=args.focal_loss_gamma if hasattr(args, 'focal_loss_gamma') else 2.0,
        ).to(self.device)

        # Get data transformation functions
        self.train_transform, self.test_transform = get_transforms()

        # Set up training or testing environment based on mode
        if not args.eval:
            self._setup_training()
        else:
            self._setup_testing()

    def _setup_training(self):
        """Set up training environment, including optimizer, scheduler, datasets, and logging"""
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
        if self.args.differential_lr and hasattr(self.model, 'get_parameter_groups'):
            # Use different learning rates for backbone and other parts
            param_groups = self.model.get_parameter_groups(backbone_lr_factor=self.args.backbone_lr_factor)
            optimizer = torch.optim.AdamW([
                {'params': group['params'], 'lr': self.args.lr * group['lr_factor']}
                for group in param_groups
            ], weight_decay=self.args.weight_decay, betas=(0.9, 0.999), eps=1e-8)
            
            print(f"Using differential learning rates:")
            for i, group in enumerate(param_groups):
                print(f"  Group {i+1}: {len(group['params'])} parameters, lr_factor={group['lr_factor']}")
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
        if hasattr(self.args, 'freeze_backbone_layers') and self.args.freeze_backbone_layers > 0:
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
        """Prepare training and validation datasets by randomly splitting the original dataset"""
        # Load complete dataset
        dataset = SegmentationDataset(root_dir=self.args.train_data_dir)
        
        # Split dataset into training and validation sets
        val_size = int(len(dataset) * 0.1)  # Use 10% of data as validation set
        train_size = len(dataset) - val_size
        self.train_dataset, self.valid_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),  # Fixed seed for reproducibility
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
            shuffle=True,  # Shuffle data order
            num_workers=16,  # Use 8 worker processes to load data
            collate_fn=collate_fn,  # Custom batch collation function
            pin_memory=True,
        )

        # Validation data loader
        self.val_loader = DataLoader(
            self.valid_dataset,
            batch_size=1,  # Use batch size of 1 for validation
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
        # Initialize gradient scaler for mixed precision training
        scaler = GradScaler()
        
        print(f"Starting training: {self.args.epochs} epochs, batch size {self.args.batch_size}")
        print(f"Learning rate: {self.args.lr}, weight decay: {self.args.weight_decay}")

        # Main training loop
        for epoch in range(1, self.args.epochs + 1):
            # Train one epoch
            self.model.train()
            train_metrics = self._train_epoch(epoch, scaler)
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()

            # Perform validation periodically
            if epoch % val_every_epochs == 0:
                val_map = self._validate_epoch(epoch) if self.val_loader else 0.0
                self._update_best_model(val_map, epoch)
                self._log_metrics(epoch, train_metrics, val_map)

            # Clear GPU cache
            torch.cuda.empty_cache()

        # Save final training curves
        self._save_training_plots()
        
        # Finish training
        self.writer.close()
        print(f"Training complete. Best mAP: {self.best_map:.4f}")
        print(f"Training plots saved to {os.path.join(self.args.output_dir, 'plots')}")
        print(f"Visualization results saved to {os.path.join(self.args.output_dir, 'viz')}")

    def _train_epoch(self, epoch, scaler):
        """
        Train one epoch
        
        Args:
            epoch (int): Current epoch
            scaler (GradScaler): Gradient scaler
            
        Returns:
            dict: Dictionary of training metrics
        """
        # Initialize metrics dictionary
        metrics = {
            "cls_loss": 0.0,  # Classification loss
            "box_loss": 0.0,  # Bounding box regression loss
            "obj_loss": 0.0,  # Objectness loss
            "rpn_box_loss": 0.0,  # RPN bounding box regression loss
            "mask_loss": 0.0,  # Mask segmentation loss
            "total_loss": 0.0,  # Total loss
        }
        total_batches = len(self.train_loader)

        # Iterate through training data batches
        for images, targets in tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.args.epochs}",
            unit="batch",
            leave=False,
            ncols=80,
        ):
            # Move data to specified device
            images = [image.to(self.device) for image in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            # Calculate loss and update parameters
            self.optimizer.zero_grad()
            loss_dict = self._compute_loss(images, targets, scaler)
            self._update_metrics(metrics, loss_dict)

        # Return average metrics
        return {k: v / total_batches for k, v in metrics.items()}

    def _compute_loss(self, images, targets, scaler):
        """
        Calculate loss and perform backpropagation
        
        Args:
            images: List of images
            targets: List of target dictionaries
            scaler: Gradient scaler
            
        Returns:
            dict: Loss dictionary
        """
        # Use automatic mixed precision
        with autocast(device_type=self.device.type, enabled=True):
            loss_dict = self.model(images, targets)
            total_loss = sum(loss_dict.values())

        # Use gradient scaler for backpropagation and parameter update
        scaler.scale(total_loss).backward()
        scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # Gradient clipping
        scaler.step(self.optimizer)
        scaler.update()

        return loss_dict

    def _update_metrics(self, metrics, loss_dict):
        """
        Update training metrics
        
        Args:
            metrics (dict): Metrics dictionary
            loss_dict (dict): Loss dictionary
        """
        # Accumulate individual loss values
        metrics["cls_loss"] += loss_dict["loss_classifier"].item()
        metrics["box_loss"] += loss_dict["loss_box_reg"].item()
        metrics["obj_loss"] += loss_dict["loss_objectness"].item()
        metrics["rpn_box_loss"] += loss_dict["loss_rpn_box_reg"].item()
        metrics["mask_loss"] += loss_dict["loss_mask"].item()
        metrics["total_loss"] += sum(loss_dict.values()).item()

    def _validate_epoch(self, epoch):
        """
        Validate one epoch
        
        Args:
            epoch (int): Current epoch
            
        Returns:
            float: Validation set mAP
        """
        self.model.eval()
        preds = []
        coco_gt = self._create_coco_gt()
        
        # Collect validation examples for visualization
        val_examples = []
        num_viz_examples = 10  # Number of examples to visualize

        # Perform forward propagation without calculating gradients
        with torch.no_grad():
            # Iterate through validation data batches
            for idx, (images, targets) in enumerate(tqdm(
                self.val_loader,
                desc=f"Val {epoch}/{self.args.epochs}",
                unit="batch",
                leave=False,
                ncols=80,
                disable=None,
            )):
                # Move images to specified device
                images = [image.to(self.device) for image in images]
                targets = [{k: v for k, v in t.items()} for t in targets]
                image_ids = [t["image_id"].item() for t in targets]

                # Get predictions
                predictions = self.model(images)
                self._process_predictions(predictions, targets, image_ids, preds, coco_gt)
                
                # Collect examples for visualization (first 10 samples)
                if idx < num_viz_examples:
                    # Store original image, ground truth, and prediction
                    val_examples.append({
                        'image': images[0].cpu(),  # First image in batch (batch size is 1 for validation)
                        'target': targets[0],      # Ground truth
                        'prediction': predictions[0]  # Model prediction
                    })

        # Calculate mAP
        print("\rCalculating mAP...", end="", flush=True)
        val_map = self._compute_map(preds, coco_gt)
        print(f"\rValidation mAP: {val_map:.4f}      ", flush=True)
        
        # Visualize segmentation results
        if epoch % 5 == 0 or epoch == self.args.epochs:  # Visualize every 5 epochs and last epoch
            self._visualize_segmentation(val_examples, epoch)
        
        return val_map

    def _create_coco_gt(self):
        """
        Create COCO format ground truth
        
        Returns:
            dict: COCO format ground truth dictionary
        """
        return {
            "images": [],
            "annotations": [],
            "categories": [{"id": i, "name": f"class_{i}"} for i in range(1, 5)],
        }

    def _process_predictions(self, predictions, targets, image_ids, preds, coco_gt):
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
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".json") as pred_f, \
             tempfile.NamedTemporaryFile(mode="w+", suffix=".json") as gt_f:
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
                print(f"Average Precision (AP) @ IoU=0.50: {coco_eval.stats[0]:.4f}")
                print(f"Average Precision (AP) @ IoU=0.75: {coco_eval.stats[1]:.4f}")
                print(f"Average Precision (AP) @ IoU=0.50:0.95: {coco_eval.stats[2]:.4f}")
                print(f"Average Recall (AR) @ IoU=0.50:0.95: {coco_eval.stats[8]:.4f}")
                
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
        total_loss = train_metrics["total_loss"]
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

    def _visualize_segmentation(self, examples, epoch):
        """
        Visualize segmentation results for validation examples
        
        Args:
            examples (list): List of examples containing image, target, and prediction
            epoch (int): Current epoch
        """
        print("Visualizing segmentation results...")
        
        for i, example in enumerate(examples):
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
            
            # Get predicted masks, scores, and labels
            pred_masks = example['prediction']['masks'].cpu().numpy()
            pred_scores = example['prediction']['scores'].cpu().numpy()
            pred_labels = example['prediction']['labels'].cpu().numpy()
            
            # Create a figure with 3 subplots: original image, ground truth, prediction
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Original image
            axes[0].imshow(img)
            axes[0].set_title("Original Image")
            axes[0].axis('off')
            
            # Ground truth
            axes[1].imshow(img)
            gt_combined_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            for mask, label in zip(gt_masks, gt_labels):
                mask_binary = mask > 0.5
                gt_combined_mask[mask_binary] = label
            
            axes[1].imshow(gt_combined_mask, alpha=0.5, cmap=CLASS_COLORS, vmin=0, vmax=4)
            axes[1].set_title("Ground Truth")
            axes[1].axis('off')
            
            # Prediction
            axes[2].imshow(img)
            pred_combined_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            
            # Only show predictions with score above threshold
            for mask, score, label in zip(pred_masks, pred_scores, pred_labels):
                if score > 0.5:  # Threshold for visualization
                    mask_binary = mask[0] > 0.5
                    pred_combined_mask[mask_binary] = label
            
            axes[2].imshow(pred_combined_mask, alpha=0.5, cmap=CLASS_COLORS, vmin=0, vmax=4)
            axes[2].set_title(f"Prediction (Epoch {epoch})")
            axes[2].axis('off')
            
            # Save the figure
            plt.tight_layout()
            plt.savefig(os.path.join(self.args.output_dir, "viz", f"val_sample_{i}_epoch_{epoch}.png"))
            plt.close(fig)
        
        print(f"Saved {len(examples)} visualization samples to {os.path.join(self.args.output_dir, 'viz')}")

    def _save_training_plots(self):
        """
        Save plots of training metrics
        """
        if not self.epoch_nums:
            return  # No data to plot yet
            
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot loss
        ax1.plot(self.epoch_nums, self.epoch_losses, 'b-', marker='o')
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        # Plot mAP
        ax2.plot(self.epoch_nums, self.epoch_maps, 'r-', marker='o')
        ax2.set_title('Validation mAP')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('mAP')
        ax2.grid(True)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.args.output_dir, "plots", "training_curves.png"))
        plt.close(fig)
        
        # Also save the data as CSV for later analysis
        import csv
        with open(os.path.join(self.args.output_dir, "plots", "training_metrics.csv"), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Epoch', 'Loss', 'mAP'])
            for i in range(len(self.epoch_nums)):
                writer.writerow([self.epoch_nums[i], self.epoch_losses[i], self.epoch_maps[i]])
                
        print(f"Saved training curves to {os.path.join(self.args.output_dir, 'plots')}")

    @torch.no_grad()
    def test(self, load_weights=True):
        """
        Evaluate model on test set
        
        Args:
            load_weights (bool): Whether to load weights
        """
        # Set to evaluation mode
        self.model.eval()
        
        # Load weights
        if load_weights:
            self.load_weights()
            print(f"Loaded weights: {self.args.ckpt_name}")
            
        # Force evaluation mode on the model
        self.model.train(False)
        for module in self.model.modules():
            if hasattr(module, 'training'):
                module.training = False

        # Collect prediction results
        preds = []
        for images, idx in tqdm(
            self.test_loader, 
            desc="Testing", 
            unit="batch", 
            leave=False,
            ncols=80,
        ):
            # Move images to specified device
            images = [image.to(self.device) for image in images]
            
            # Make predictions without targets
            with torch.no_grad():
                predictions = self.model(images)

            # Process each image's prediction results
            for i, pred in zip(idx, predictions):
                for j in range(len(pred["masks"])):
                    x_min, y_min, x_max, y_max = pred["boxes"][j].cpu().tolist()
                    mask = pred["masks"][j].cpu().numpy()[0]
                    binary_mask = (mask >= 0.5).astype(np.uint8)

                    preds.append({
                        "image_id": i,
                        "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                        "score": pred["scores"][j].item(),
                        "category_id": pred["labels"][j].item(),
                        "segmentation": encode_mask(binary_mask),
                    })

        print(f"Generated {len(preds)} prediction results. Saving...")
        self._save_test_results(preds)

    def _save_test_results(self, preds):
        """
        Save test results
        
        Args:
            preds: Prediction result list
        """
        # Save prediction results to JSON file
        with open("test-results.json", "w") as f:
            json.dump(preds, f, indent=4)        
        # Use subprocess silently to create zip file
        os.system("zip result.zip test-results.json")
        print(f"Saved {len(preds)} prediction results to result.zip")

    def save_weights(self):
        """
        Save model weights
        """
        # Determine save path
        ckpt_path = os.path.join(self.args.output_dir, "best.pth")

        # Save model state dictionary and configuration parameters
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': {
                'backbone': self.args.backbone,
                'num_classes': self.args.num_classes,
                'box_score_thresh': self.args.box_score_thresh,
                'mask_head_type': self.args.mask_head_type,
                'use_transformer': self.args.use_transformer,
                'transformer_layers': self.args.transformer_layers,
                'transformer_dim': self.args.transformer_dim,
                'transformer_heads': self.args.transformer_heads,
                'use_focal_loss': self.args.use_focal_loss if hasattr(self.args, 'use_focal_loss') else False,
                'focal_loss_alpha': self.args.focal_loss_alpha if hasattr(self.args, 'focal_loss_alpha') else 0.25,
                'focal_loss_gamma': self.args.focal_loss_gamma if hasattr(self.args, 'focal_loss_gamma') else 2.0,
                'box_detections_per_img': self.args.box_detections_per_img if hasattr(self.args, 'box_detections_per_img') else 200,
                'rpn_pre_nms_top_n_train': self.args.rpn_pre_nms_top_n_train if hasattr(self.args, 'rpn_pre_nms_top_n_train') else 2000,
                'rpn_post_nms_top_n_train': self.args.rpn_post_nms_top_n_train if hasattr(self.args, 'rpn_post_nms_top_n_train') else 1000,
                'rpn_pre_nms_top_n_test': self.args.rpn_pre_nms_top_n_test if hasattr(self.args, 'rpn_pre_nms_top_n_test') else 1000,
                'rpn_post_nms_top_n_test': self.args.rpn_post_nms_top_n_test if hasattr(self.args, 'rpn_post_nms_top_n_test') else 500,
                'rpn_nms_thresh': self.args.rpn_nms_thresh if hasattr(self.args, 'rpn_nms_thresh') else 0.7,
                'roi_nms_thresh': self.args.roi_nms_thresh if hasattr(self.args, 'roi_nms_thresh') else 0.5,
            },
            'best_map': self.best_map,
            'optimizer_state_dict': self.optimizer.state_dict() if hasattr(self, 'optimizer') else None,
            'scheduler_state_dict': self.scheduler.state_dict() if hasattr(self, 'scheduler') else None,
        }
        torch.save(checkpoint, ckpt_path)
        print(f"Saved model weights and configuration to {ckpt_path}")

    def load_weights(self):
        """Load model weights and configuration"""
        # Check if checkpoint path is provided
        if self.args.ckpt_name is None:
            raise ValueError("Please use --ckpt_name to specify checkpoint path in evaluation mode")
        
        # Load checkpoint
        ckpt = torch.load(self.args.ckpt_name)
        
        # Check if the checkpoint contains configuration
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt and 'config' in ckpt:
            # If configuration exists, update the args with saved configuration
            config = ckpt['config']
            print("Loading model configuration from checkpoint:")
            for key, value in config.items():
                if hasattr(self.args, key):
                    setattr(self.args, key, value)
                    print(f"  {key}: {value}")
            
            # Recreate the model with loaded configuration
            self.model = CustomMaskRCNN(
                num_classes=self.args.num_classes,
                box_score_thresh=self.args.box_score_thresh,
                backbone=self.args.backbone,
                mask_head_type=self.args.mask_head_type,
                use_transformer=self.args.use_transformer,
                transformer_layers=self.args.transformer_layers,
                transformer_dim=self.args.transformer_dim,
                transformer_heads=self.args.transformer_heads,
                use_focal_loss=self.args.use_focal_loss if hasattr(self.args, 'use_focal_loss') else False,
                focal_loss_alpha=self.args.focal_loss_alpha if hasattr(self.args, 'focal_loss_alpha') else 0.25,
                focal_loss_gamma=self.args.focal_loss_gamma if hasattr(self.args, 'focal_loss_gamma') else 2.0,
            ).to(self.device)
            
            # Load the state dict
            self.model.load_state_dict(ckpt['model_state_dict'])
            
            # Load best mAP if available
            if 'best_map' in ckpt:
                self.best_map = ckpt['best_map']
                print(f"Previous best mAP: {self.best_map:.4f}")
        else:
            # Legacy checkpoint (just the state dict)
            self.model.load_state_dict(ckpt)
