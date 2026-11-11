# Third-party notices

The original project code uses [Apache-2.0](LICENSE) where the project
contributors can grant it. Upstream code, model derivatives and datasets
carry their own terms. The
[initializer catalog](configs/upstream_initialization.json) records the
publishers and exact pretrained variants used for training.

| Component | Upstream source and terms |
| --- | --- |
| Co-DETR / Co-DINO | [Pinned fork](https://github.com/ljc-1222/Co-DETR/tree/a3a9d20b7b2e19ad6e140590a6e3d8f2d3c88e9b), [MIT](licenses/Co-DETR-MIT.txt). The [Objects365 initializer](https://huggingface.co/zongzhuofan/co-detr-vit-large-objects365/blob/adb854bf5fa409d15d45452f906fbccf6e60fdab/README.md) declares [Apache-2.0](licenses/Apache-2.0.txt). |
| TinyFormer XL-PBM | [Pinned source](https://github.com/mmpmmpmmpjosh/TinyFormer/tree/9075d9fdf5ce989ca3c7e830a780dfc7fbbe4ec1), [Apache-2.0 root license](licenses/TinyFormer-root-LICENSE.txt), with a [DINOv3 adapter](https://github.com/mmpmmpmmpjosh/TinyFormer/blob/9075d9fdf5ce989ca3c7e830a780dfc7fbbe4ec1/engine/backbone/dinov3_adapter.py) under the [DINOv3 Agreement](licenses/DINOv3-LICENSE.md). We apply the repository-level Apache grant to the authors' contributions in their [model-zoo file](https://drive.google.com/drive/folders/1sW5AfpNvz12yKcqoUYj3uk6tUUBruNvW) and retain the DINOv3 Agreement for its base. |
| RF-DETR Legacy Large | [RF-DETR 1.9.0](https://github.com/roboflow/rf-detr/tree/1.9.0), [Apache-2.0](licenses/RF-DETR-Apache-2.0.txt). The [legacy COCO weights](https://github.com/roboflow/rf-detr/blob/1.3.0/README.md#license) are Apache-2.0. The recipe uses `RFDETRLargeDeprecated` and `rf-detr-large.pth`. |
| Ultralytics YOLO26 | [Pinned source](https://github.com/ultralytics/ultralytics/tree/dad7bb4534c95021bc14969ab25d77b77c4efdc3) and [AGPL-3.0](licenses/Ultralytics-AGPL-3.0.txt). [Ultralytics terms](https://www.ultralytics.com/license) cover its pretrained and fine-tuned models. The [local training adapter](training/yolo_data.py) retains its AGPL-3.0 header. |
| DINOv3 ViT-H+ and ViT-L | [Pinned source](https://github.com/facebookresearch/dinov3/tree/6876159a11b4df116f30f667f8c9888617df0751) and [DINOv3 Agreement](licenses/DINOv3-LICENSE.md). The Agreement applies to code, weights and derivatives, including the complete H+ state with unmerged LoRA and the ViT-L crop classifier. |
| SigLIP2 image tower | The [pinned timm model card](https://huggingface.co/timm/vit_large_patch16_siglip_384.v2_webli/blob/62eadefe185d74bd40be43b62d17a135ac417404/README.md) declares Apache-2.0 for this checkpoint. [timm 1.0.20](https://github.com/huggingface/pytorch-image-models/tree/v1.0.20) uses [Apache-2.0](licenses/timm-Apache-2.0.txt). |

Other pinned dependencies include [NVIDIA NVML bindings](https://pypi.org/project/nvidia-ml-py/13.615.71/)
(BSD-3-Clause), [Ultralytics Platform SDK](https://github.com/ultralytics/sdk/blob/69a17458fd45841cfb44d6217afc95b9f3453395/LICENSE)
(AGPL-3.0-only), [Ultralytics THOP](https://github.com/ultralytics/thop/blob/39438aabf253157d90ecd3a3b8dde4e1e5f7beed/LICENSE)
(AGPL-3.0) and [Ninja](https://pypi.org/project/ninja/1.13.2/) (Apache-2.0).
Retain file-level notices and applicable license texts when redistributing
these libraries.

## Distributing model files

For Apache-covered components, retain the license, attribution and applicable
NOTICE material, and mark modifications. Include the full DINOv3 Agreement
with derived TinyFormer and DCR weights. Follow Meta's
[native weight access process](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)
for new training and acknowledge DINOv3 in research publications using it.

Supply matching corresponding source when distributing the YOLO26 artifact
under AGPL-3.0. This includes the pinned Ultralytics source, applicable project
training and inference code, [the exporter](training/export_yolo26.py),
recipes, dependencies, and modification notices. Provide durable access to
that source alongside the download. The distribution must follow the
applicable license or a separate agreement with the rightsholder.

Build the complete archive from a committed checkout with initialized
submodules:

```bash
python scripts/release_archive.py build --output-dir outputs/source-release
```

## Data

NAMR's [MDImageNet release announcement](https://www.namr.gov.tw/ch/home.jsp?dataserno=202607300001&id=36&mcustomize=news_view.jsp&parentpath=0%2C6)
identifies [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). For
covered images and derivatives, retain the provider credit, source and license
and describe any changes. Follow the [dataset service terms](https://mdimagenet.namr.gov.tw/namrOceanTrash/service)
and file-specific notices. SAM's [software and model terms](https://github.com/facebookresearch/sam3/blob/main/LICENSE)
are separate from the rights in donor image pixels. Objects365 has its own
[dataset terms](https://www.objects365.org/download.html).

[Example image credits](assets/examples/README.md) · [Training data preparation](docs/training.md)

## Method references

- Solovyev, Wang and Gabruseva (2021). [Weighted Boxes Fusion](https://arxiv.org/abs/1910.13302).
- Cheng et al. (ECCV 2018). [Revisiting RCNN](https://openaccess.thecvf.com/content_ECCV_2018/html/Bowen_Cheng_Revisiting_RCNN_On_ECCV_2018_paper.html).
- Bodla et al. (ICCV 2017). [Soft-NMS](https://arxiv.org/abs/1704.04503).
