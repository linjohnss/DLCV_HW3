import os
import random
import argparse
import numpy as np
import torch

from utils.trainer import Trainer


def set_seed(seed=20250507):
    """
    Set random seed to ensure reproducibility

    Args:
        seed (int): Random seed value, default is 42
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def parse_args():
    """
    Parse command line arguments

    Returns:
        argparse.Namespace: Object containing all command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Instance Segmentation Model Training and Evaluation"
    )

    # Model configuration
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument(
        "--backbone",
        type=str,
        default="default",
        help="Backbone network type (default, resnet50, convnext_tiny, "
             "convnext_small, convnext_base, convnext_large)"
    )
    model_group.add_argument(
        "--num_classes",
        type=int,
        default=5,
        help="Number of classes (including background)"
    )
    model_group.add_argument(
        "--box_score_thresh",
        type=float,
        default=0.5,
        help="Bounding box confidence threshold"
    )
    model_group.add_argument(
        "--mask_head_type",
        type=str,
        default="default",
        help="Mask head type (default, deep, wider)"
    )
    model_group.add_argument(
        "--use_transformer",
        action="store_true",
        default=False,
        help="Use transformer decoder in the mask head"
    )
    model_group.add_argument(
        "--transformer_layers",
        type=int,
        default=2,
        help="Number of transformer layers when using transformer decoder"
    )
    model_group.add_argument(
        "--transformer_dim",
        type=int,
        default=256,
        help="Dimension of transformer layers"
    )
    model_group.add_argument(
        "--transformer_heads",
        type=int,
        default=8,
        help="Number of attention heads in transformer"
    )

    # Detection parameters
    detection_group = parser.add_argument_group('Detection Parameters')
    detection_group.add_argument(
        "--box_detections_per_img",
        type=int,
        default=200,
        help="Maximum number of detections per image"
    )
    detection_group.add_argument(
        "--rpn_pre_nms_top_n_train",
        type=int,
        default=2000,
        help="Number of proposals before NMS in training"
    )
    detection_group.add_argument(
        "--rpn_post_nms_top_n_train",
        type=int,
        default=1000,
        help="Number of proposals after NMS in training"
    )
    detection_group.add_argument(
        "--rpn_pre_nms_top_n_test",
        type=int,
        default=1000,
        help="Number of proposals before NMS in testing"
    )
    detection_group.add_argument(
        "--rpn_post_nms_top_n_test",
        type=int,
        default=500,
        help="Number of proposals after NMS in testing"
    )
    detection_group.add_argument(
        "--rpn_nms_thresh",
        type=float,
        default=0.7,
        help="NMS threshold for RPN"
    )
    detection_group.add_argument(
        "--roi_nms_thresh",
        type=float,
        default=0.5,
        help="NMS threshold for ROI heads"
    )

    # Training configuration
    train_group = parser.add_argument_group('Training Configuration')
    train_group.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs"
    )
    train_group.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size"
    )
    train_group.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    train_group.add_argument(
        "--weight_decay",
        type=float,
        default=1e-2,
        help="Weight decay"
    )
    train_group.add_argument(
        "--use_focal_loss",
        action="store_true",
        default=False,
        help="Use Focal Loss for classification"
    )
    train_group.add_argument(
        "--focal_loss_alpha",
        type=float,
        default=0.25,
        help="Alpha parameter for Focal Loss"
    )
    train_group.add_argument(
        "--focal_loss_gamma",
        type=float,
        default=2.0,
        help="Gamma parameter for Focal Loss"
    )
    train_group.add_argument(
        "--differential_lr",
        action="store_true",
        default=False,
        help="Use different learning rates for backbone and heads"
    )
    train_group.add_argument(
        "--backbone_lr_factor",
        type=float,
        default=0.1,
        help="Learning rate factor for backbone "
             "(applied when differential_lr is True)"
    )
    train_group.add_argument(
        "--freeze_backbone_layers",
        type=int,
        default=0,
        help="Number of backbone layers to freeze "
             "(0 means no freezing)"
    )

    # Dataset configuration
    data_group = parser.add_argument_group('Dataset Configuration')
    data_group.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Data root directory"
    )

    # Logging and checkpointing configuration
    log_group = parser.add_argument_group('Logging and Checkpointing')
    log_group.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Output directory for checkpoints and logs"
    )

    # Evaluation configuration
    eval_group = parser.add_argument_group('Evaluation Configuration')
    eval_group.add_argument(
        "--eval",
        action="store_true",
        default=False,
        help="Enable evaluation mode"
    )
    eval_group.add_argument(
        "--ckpt_name",
        type=str,
        help="Checkpoint file path for evaluation"
    )

    # Parse arguments
    args = parser.parse_args()

    # Set device
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set fixed paths
    args.train_data_dir = os.path.join(args.data_dir, "train")
    args.test_data_dir = os.path.join(args.data_dir, "test_release")
    args.json_map_imgname_to_id = os.path.join(
        args.data_dir,
        "test_image_name_to_ids.json"
    )

    return args


def main():
    """
    Main function: sets random seed, parses arguments, initializes trainer
    and performs training or evaluation
    """
    # Set random seed for reproducibility
    set_seed()

    # Parse command line arguments
    args = parse_args()

    # Print initial configuration info
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Backbone: {args.backbone}")
    print(f"Mask head: {args.mask_head_type}")
    if args.use_transformer:
        print(
            f"Using transformer: layers={args.transformer_layers}, "
            f"heads={args.transformer_heads}, dim={args.transformer_dim}"
        )
    if args.use_focal_loss:
        print(
            f"Using Focal Loss: alpha={args.focal_loss_alpha}, "
            f"gamma={args.focal_loss_gamma}"
        )
    if args.differential_lr:
        print(
            f"Using differential learning rates: "
            f"backbone_factor={args.backbone_lr_factor}"
        )
    if args.freeze_backbone_layers > 0:
        print(f"Freezing {args.freeze_backbone_layers} backbone layers")
    print(f"Mode: {'Evaluation' if args.eval else 'Training'}")

    # Initialize trainer
    trainer = Trainer(args)

    # Print final configuration (in case it was loaded from checkpoint)
    if args.eval and hasattr(args, 'backbone') and args.backbone != "default":
        print("\nFinal configuration after loading checkpoint:")
        print(f"Backbone: {args.backbone}")
        print(f"Mask head: {args.mask_head_type}")
        if args.use_transformer:
            print(
                f"Using transformer: layers={args.transformer_layers}, "
                f"heads={args.transformer_heads}, dim={args.transformer_dim}"
            )

    # Perform training or evaluation based on mode
    if args.eval:
        print("Starting model evaluation...")
        trainer.test()
    else:
        print("Starting model training...")
        trainer.train()
        print("Training complete!")


if __name__ == "__main__":
    main()
