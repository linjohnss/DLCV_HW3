import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torchvision.models.detection import (
    MaskRCNN,
    maskrcnn_resnet50_fpn_v2,
    MaskRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.models import (
    convnext_tiny, ConvNeXt_Tiny_Weights,
    convnext_small, ConvNeXt_Small_Weights,
    convnext_base, ConvNeXt_Base_Weights,
    convnext_large, ConvNeXt_Large_Weights,
)
from torchvision.ops.feature_pyramid_network import (
    FeaturePyramidNetwork,
    LastLevelMaxPool,
)
from torchvision.models.detection.mask_rcnn import (
    MaskRCNNHeads,
    MaskRCNNPredictor,
)


class ConvNeXtFPN(nn.Module):
    """
    Custom FPN implementation for ConvNeXt backbone
    """
    def __init__(self, backbone, backbone_name="convnext_tiny"):
        super().__init__()
        self.backbone = backbone

        # Define the feature channels for each level based on backbone type
        if backbone_name == "convnext_tiny":
            # ConvNeXt-Tiny feature channels
            self.in_channels_list = [96, 192, 384, 768]
        elif backbone_name == "convnext_small":
            # ConvNeXt-Small feature channels
            self.in_channels_list = [96, 192, 384, 768]
        elif backbone_name == "convnext_base":
            # ConvNeXt-Base feature channels
            self.in_channels_list = [128, 256, 512, 1024]
        elif backbone_name == "convnext_large":
            # ConvNeXt-Large feature channels
            self.in_channels_list = [192, 384, 768, 1536]
        else:
            raise ValueError(f"Unsupported ConvNeXt variant: {backbone_name}")

        out_channels = 256

        # Create FPN
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=self.in_channels_list,
            out_channels=out_channels,
            extra_blocks=LastLevelMaxPool(),
        )

        # Store the output channels for later use
        self.out_channels = out_channels

    def forward(self, x):
        # Get intermediate features from backbone
        features = []
        x = self.backbone.features[0](x)  # Initial stem
        features.append(x)  # 1/4 scale

        # Get features from each stage
        for stage in self.backbone.features[1:]:
            x = stage(x)
            features.append(x)

        # Select the correct feature layers for FPN
        # We need features at 1/4, 1/8, 1/16, and 1/32 scales
        selected_features = {
            '0': features[0],  # 1/4 scale (96 channels)
            '1': features[2],  # 1/8 scale (192 channels)
            '2': features[4],  # 1/16 scale (384 channels)
            '3': features[6],  # 1/32 scale (768 channels)
        }

        # Apply FPN
        return self.fpn(selected_features)


class TransformerMaskDecoder(nn.Module):
    """
    Transformer decoder for mask prediction

    This module uses transformer layers to refine mask features before final
    prediction
    """
    def __init__(
        self,
        in_channels,
        dim_feedforward=2048,
        nhead=8,
        num_layers=2,
        dropout=0.1
    ):
        super().__init__()

        # Initial convolution to project input features
        self.conv_in = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1
        )

        # Create positional encoding
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, in_channels, 14, 14)
        )
        nn.init.normal_(self.pos_embedding, std=0.02)

        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_channels,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True
        )

        # Transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )

        # Output convolution
        self.conv_out = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):
        # Apply initial convolution
        x = self.conv_in(x)

        # Add positional encoding
        x = x + self.pos_embedding

        # Reshape for transformer: [B, C, H, W] -> [B, H*W, C]
        batch_size, channels, height, width = x.shape
        x_flat = x.flatten(2).permute(0, 2, 1)

        # Apply transformer encoder
        x_transformed = self.transformer_encoder(x_flat)

        # Reshape back to 2D: [B, H*W, C] -> [B, C, H, W]
        x_2d = x_transformed.permute(0, 2, 1).view(
            batch_size, channels, height, width
        )

        # Apply output convolution
        out = self.conv_out(x_2d)

        return out


class EnhancedMaskRCNNPredictor(nn.Module):
    """
    Enhanced Mask R-CNN predictor with more layers and transformer option
    """
    def __init__(
        self,
        in_channels,
        dim_inner=256,
        num_classes=2,
        mask_head_type="default",
        use_transformer=False,
        transformer_layers=2,
        transformer_heads=8
    ):
        super().__init__()

        # Choose mask head type
        if mask_head_type == "default":
            # Default mask head with single hidden layer
            layers = [
                nn.Conv2d(
                    in_channels,
                    dim_inner,
                    kernel_size=3,
                    stride=1,
                    padding=1
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    dim_inner,
                    dim_inner,
                    kernel_size=3,
                    stride=1,
                    padding=1
                ),
                nn.ReLU(inplace=True),
            ]
        elif mask_head_type == "deep":
            # Deeper mask head with multiple hidden layers
            layers = [
                nn.Conv2d(
                    in_channels,
                    dim_inner,
                    kernel_size=3,
                    stride=1,
                    padding=1
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    dim_inner,
                    dim_inner,
                    kernel_size=3,
                    stride=1,
                    padding=1
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    dim_inner,
                    dim_inner,
                    kernel_size=3,
                    stride=1,
                    padding=1
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    dim_inner,
                    dim_inner,
                    kernel_size=3,
                    stride=1,
                    padding=1
                ),
                nn.ReLU(inplace=True),
            ]
        elif mask_head_type == "wider":
            # Wider mask head with more channels
            wider_dim = dim_inner * 2
            layers = [
                nn.Conv2d(
                    in_channels,
                    wider_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    wider_dim,
                    wider_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    wider_dim,
                    dim_inner,
                    kernel_size=3,
                    stride=1,
                    padding=1
                ),
                nn.ReLU(inplace=True),
            ]
        else:
            raise ValueError(f"Unsupported mask head type: {mask_head_type}")

        self.mask_fcn = nn.Sequential(*layers)

        # Add transformer if requested
        self.use_transformer = use_transformer
        if use_transformer:
            self.transformer = TransformerMaskDecoder(
                in_channels=dim_inner,
                nhead=transformer_heads,
                num_layers=transformer_layers
            )

        # Final prediction layer
        self.mask_pred = nn.Conv2d(
            dim_inner,
            num_classes,
            kernel_size=1,
            stride=1
        )

    def forward(self, x):
        x = self.mask_fcn(x)

        if self.use_transformer:
            x = self.transformer(x)

        return self.mask_pred(x)


class FocalLoss(nn.Module):
    """
    Focal Loss for dense object detection

    Args:
        alpha (float): Weighting factor for the rare class
        gamma (float): Focusing parameter that adjusts the rate at which easy
            examples are down-weighted
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        """
        Calculate focal loss

        Args:
            logits (torch.Tensor): Model predictions
            targets (torch.Tensor): Target values

        Returns:
            torch.Tensor: Calculated loss
        """
        # Get probability using sigmoid
        probs = torch.sigmoid(logits)

        # Calculate BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction='none'
        )

        # Apply focal loss formula
        pt = torch.where(targets == 1, probs, 1 - probs)
        alpha_factor = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_weight = alpha_factor * (1 - pt) ** self.gamma

        # Apply weights and take mean
        loss = focal_weight * bce_loss
        return loss.mean()


class CustomMaskRCNN(nn.Module):
    """
    Custom Mask R-CNN model

    Wraps torchvision's Mask R-CNN implementation, providing an interface
    suitable for the current task
    """
    def __init__(
        self,
        num_classes=5,
        box_score_thresh=0.5,
        backbone="default",
        mask_head_type="default",
        use_transformer=False,
        transformer_layers=2,
        transformer_dim=256,
        transformer_heads=8,
        use_focal_loss=False,
        focal_loss_alpha=0.25,
        focal_loss_gamma=2.0
    ):
        """
        Initialize CustomMaskRCNN model

        Args:
            num_classes (int): Number of classes (including background)
            box_score_thresh (float): Bounding box confidence threshold
            backbone (str): Backbone network type
            mask_head_type (str): Type of mask head to use
            use_transformer (bool): Whether to use transformer decoder
            transformer_layers (int): Number of transformer layers
            transformer_dim (int): Dimension of transformer features
            transformer_heads (int): Number of attention heads
            use_focal_loss (bool): Whether to use focal loss
            focal_loss_alpha (float): Alpha parameter for focal loss
            focal_loss_gamma (float): Gamma parameter for focal loss
        """
        super().__init__()

        # Store configuration options
        self.num_classes = num_classes
        self.backbone_type = backbone
        self.mask_head_type = mask_head_type
        self.use_transformer = use_transformer
        self.transformer_layers = transformer_layers
        self.transformer_dim = transformer_dim
        self.transformer_heads = transformer_heads
        self.use_focal_loss = use_focal_loss
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma

        # Initialize model with different backbone options
        if backbone == "default" or backbone == "resnet50":
            # Use ResNet50 FPN V2 with pretrained weights
            self.model = maskrcnn_resnet50_fpn_v2(
                weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
                box_score_thresh=box_score_thresh,
                box_detections_per_img=200,
                rpn_pre_nms_top_n_train=2000,
                rpn_post_nms_top_n_train=1000,
                rpn_pre_nms_top_n_test=1000,
                rpn_post_nms_top_n_test=500,
            )

            # Replace the mask predictor with our custom one
            in_features_mask = (
                self.model.roi_heads.mask_predictor.conv5_mask.in_channels
            )
            hidden_layer = 256
            self.model.roi_heads.mask_predictor = EnhancedMaskRCNNPredictor(
                in_channels=in_features_mask,
                dim_inner=hidden_layer,
                num_classes=num_classes,
                mask_head_type=mask_head_type,
                use_transformer=use_transformer,
                transformer_layers=transformer_layers,
                transformer_heads=transformer_heads
            )

        elif backbone in [
            "convnext_tiny",
            "convnext_small",
            "convnext_base",
            "convnext_large"
        ]:
            # Initialize proper ConvNeXt variant with pretrained weights
            if backbone == "convnext_tiny":
                convnext = convnext_tiny(
                    weights=ConvNeXt_Tiny_Weights.DEFAULT
                )
            elif backbone == "convnext_small":
                convnext = convnext_small(
                    weights=ConvNeXt_Small_Weights.DEFAULT
                )
            elif backbone == "convnext_base":
                convnext = convnext_base(
                    weights=ConvNeXt_Base_Weights.DEFAULT
                )
            elif backbone == "convnext_large":
                convnext = convnext_large(
                    weights=ConvNeXt_Large_Weights.DEFAULT
                )

            # Create custom FPN backbone
            backbone_fpn = ConvNeXtFPN(
                convnext,
                backbone_name=backbone
            )

            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            anchor_generator = (
                torchvision.models.detection.anchor_utils.AnchorGenerator(
                    anchor_sizes,
                    aspect_ratios
                )
            )

            # Initialize RPN Head and ROI Heads
            rpn_head = torchvision.models.detection.rpn.RPNHead(
                backbone_fpn.out_channels,
                anchor_generator.num_anchors_per_location()[0]
            )

            # Create ROI pooler for box and mask
            box_roi_pool = torchvision.ops.MultiScaleRoIAlign(
                featmap_names=["0", "1", "2", "3"],
                output_size=7,
                sampling_ratio=2,
            )

            # Setup RoI head parameters
            box_head = torchvision.models.detection.faster_rcnn.TwoMLPHead(
                backbone_fpn.out_channels * box_roi_pool.output_size[0] ** 2,
                1024
            )

            box_predictor = (
                torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
                    1024,
                    num_classes
                )
            )

            mask_roi_pool = torchvision.ops.MultiScaleRoIAlign(
                featmap_names=["0", "1", "2", "3"],
                output_size=14,
                sampling_ratio=2,
            )

            # Choose between standard and enhanced mask head
            if mask_head_type == "default" and not use_transformer:
                # Use standard mask head
                mask_head = MaskRCNNHeads(
                    backbone_fpn.out_channels,
                    [256, 256, 256, 256],
                    1
                )

                mask_predictor = MaskRCNNPredictor(
                    256,
                    256,
                    num_classes
                )
            else:
                # Use custom mask head and predictor
                mask_layers = [256, 256, 256, 256]  # Default mask head
                if mask_head_type == "deep":
                    mask_layers = [256, 256, 256, 256, 256, 256]  # Deeper
                elif mask_head_type == "wider":
                    mask_layers = [512, 512, 256, 256]  # Wider

                mask_head = MaskRCNNHeads(
                    backbone_fpn.out_channels,
                    mask_layers,
                    1
                )

                mask_predictor = EnhancedMaskRCNNPredictor(
                    in_channels=mask_layers[-1],
                    dim_inner=transformer_dim if use_transformer else 256,
                    num_classes=num_classes,
                    mask_head_type=mask_head_type,
                    use_transformer=use_transformer,
                    transformer_layers=transformer_layers,
                    transformer_heads=transformer_heads
                )

            # Create MaskRCNN model with custom backbone
            self.model = MaskRCNN(
                backbone=backbone_fpn,
                rpn_anchor_generator=anchor_generator,
                rpn_head=rpn_head,
                box_roi_pool=box_roi_pool,
                box_head=box_head,
                box_predictor=box_predictor,
                mask_roi_pool=mask_roi_pool,
                mask_head=mask_head,
                mask_predictor=mask_predictor,
                box_score_thresh=box_score_thresh,
                box_detections_per_img=200,
                rpn_pre_nms_top_n_train=2000,
                rpn_post_nms_top_n_train=1000,
                rpn_pre_nms_top_n_test=1000,
                rpn_post_nms_top_n_test=500,
            )
        else:
            raise ValueError(f"Unsupported backbone type: {backbone}")

        # Adjust ROI head parameters
        self.model.roi_heads.score_thresh = 0.05  # Lower score threshold
        self.model.roi_heads.nms_thresh = 0.5  # Adjust NMS threshold

        # Replace the loss function with Focal Loss if requested
        if use_focal_loss:
            self._original_loss_fn = (
                self.model.roi_heads.box_predictor.focal_loss
                if hasattr(self.model.roi_heads.box_predictor, 'focal_loss')
                else None
            )
            self.model.roi_heads.box_predictor.focal_loss = FocalLoss(
                alpha=focal_loss_alpha,
                gamma=focal_loss_gamma
            )

            # Store the original loss function method for restoration if needed
            if not hasattr(
                self.model.roi_heads.box_predictor,
                '_original_forward'
            ):
                self.model.roi_heads.box_predictor._original_forward = (
                    self.model.roi_heads.box_predictor.forward
                )

                # Override the forward method to use focal loss
                def forward_with_focal_loss(box_predictor, x):
                    # Get standard class logits and box regression outputs
                    class_logits, box_regression = (
                        box_predictor._original_forward(x)
                    )
                    return class_logits, box_regression

                self.model.roi_heads.box_predictor.forward = (
                    forward_with_focal_loss.__get__(
                        self.model.roi_heads.box_predictor
                    )
                )

        # Log model parameters
        self.log_model_parameters()

    def log_model_parameters(self):
        """
        Calculate and print model parameters
        """
        model_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"Model parameters: {model_params:.2f}M")

        # Print head configuration information
        print(f"Mask head type: {self.mask_head_type}")
        if self.use_transformer:
            print(
                f"Using transformer with {self.transformer_layers} layers "
                f"and {self.transformer_heads} heads"
            )
        if self.use_focal_loss:
            print(
                f"Using Focal Loss with alpha={self.focal_loss_alpha}, "
                f"gamma={self.focal_loss_gamma}"
            )

    def forward(self, images, targets=None):
        """
        Forward pass

        Args:
            images: List of input images
            targets: List of target dictionaries, required in training mode

        Returns:
            Training mode: Dictionary of losses
            Prediction mode: List of predictions
        """
        # In training mode, targets must be provided
        if self.training:
            if targets is None:
                raise ValueError("Targets must be provided in training mode")
            return self.model(images, targets)
        else:
            # In evaluation mode, targets should be None
            return self.model(images)

    def freeze_backbone(self, layers=3, freeze=True):
        """
        Freeze specified number of backbone layers

        Args:
            layers (int): Number of layers to freeze (from the start)
            freeze (bool): Whether to freeze or unfreeze
        """
        # Special handling for different backbones
        if self.backbone_type == "default" or self.backbone_type == "resnet50":
            # For ResNet backbone
            for i, child in enumerate(self.model.backbone.body.children()):
                if i < layers:
                    for param in child.parameters():
                        param.requires_grad = not freeze
                    print(
                        f"{'Freezing' if freeze else 'Unfreezing'} "
                        f"backbone layer {i}"
                    )

        elif self.backbone_type.startswith("convnext"):
            # For ConvNeXt backbone
            backbone = self.model.backbone.backbone
            # Freeze embedding layer (first part)
            if layers > 0:
                for param in backbone.features[0].parameters():
                    param.requires_grad = not freeze
                print(
                    f"{'Freezing' if freeze else 'Unfreezing'} "
                    f"backbone embedding layer"
                )

            # Freeze specific stages
            stages = len(backbone.features) - 1  # Exclude the embedding layer
            freeze_stages = min(layers - 1, stages)  # Adjust freeze count

            for i in range(1, freeze_stages + 1):
                for param in backbone.features[i].parameters():
                    param.requires_grad = not freeze
                print(
                    f"{'Freezing' if freeze else 'Unfreezing'} "
                    f"backbone stage {i}"
                )

    def freeze_fpn(self, freeze=True):
        """
        Freeze or unfreeze Feature Pyramid Network (FPN) parameters

        Args:
            freeze (bool): Whether to freeze the FPN
        """
        for param in self.model.backbone.fpn.parameters():
            param.requires_grad = not freeze

    def get_parameter_groups(self, backbone_lr_factor=0.1):
        """
        Set different learning rates for different parameter groups

        Args:
            backbone_lr_factor (float): Learning rate factor for backbone

        Returns:
            list: List of parameter groups
        """
        # Divide parameters into backbone parameters and other parameters
        backbone_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "backbone" in name:
                    backbone_params.append(param)
                else:
                    other_params.append(param)

        return [
            {"params": backbone_params, "lr_factor": backbone_lr_factor},
            {"params": other_params, "lr_factor": 1.0}
        ]
