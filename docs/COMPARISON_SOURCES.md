# Learned comparison models

The identifiers below map to the implementations in `compare/deep_learning/`. These are local, range-conditioned depth-regression adaptations, not claims of exact reproduction of the source training recipes.

| Identifier | Architecture/source | Local adaptation |
|---|---|---|
| `dlsim_attention_unet` | Yokoya et al., [DLSIM (IEEE TGRS)](https://doi.org/10.1109/TGRS.2020.3035469); [official code](https://github.com/nyokoya/dlsim) | Attention U-Net-style depth regressor using the common ten-channel S1/terrain/range schema. |
| `dlsim_linknet` | [DLSIM paper](https://doi.org/10.1109/TGRS.2020.3035469); [official code](https://github.com/nyokoya/dlsim) | LinkNet-style depth regressor using the common ten-channel schema. |
| `resnet18_depth_regression` | [Torchvision ResNet18](https://pytorch.org/vision/stable/models/resnet.html) | Randomly initialized ResNet18 encoder and local full-resolution depth decoder; no ImageNet weights. |
| `unet_depth_regression` | [U-Net architecture reference](https://github.com/qubvel-org/segmentation_models.pytorch) | Local U-Net depth regressor with the common input and conditional-positive output. |
| `unetplusplus_depth_regression` | [U-Net++ architecture reference](https://github.com/qubvel-org/segmentation_models.pytorch) | Local nested-decoder U-Net++ depth regressor with the common input and conditional-positive output. |

All learned methods use the shared objective and training settings defined in `configs/base/unified_protocol.xml` and `configs/base/base.xml`. The selected source architecture does not imply that its original paper used this dataset, input schema, or loss.
