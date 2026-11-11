# Training

Set up the environment in the [quick start](../README.md#quick-start) and run
these commands from the repository root. The [fixed recipes](../training/final_recipes.yaml)
use the full labeled corpus. Evaluate on a separate held-out dataset. Use
your own masks and detector proposals to build the donor bank and crop records.

## Prepare MDImageNet

Download the labeled [NAMR MDImageNet v1.0 release](https://mdimagenet.namr.gov.tw/namrOceanTrash/dataset/detail/JQdYLLKk9Qelnk6xAkqeDQ%3D%3D).
Keep `metadata_image_train.csv`, `metadata_annotation_train.csv`,
`data_NAMR33.yaml` and `train/images/` in their original layout.

```bash
python -m training.prepare_data \
  --namr-root inputs/Marine_Debris_Image_Dataset --output-root data

images=data/images/train_all
gt=data/annotations/instances_train_all
```

This writes `data/data.yaml`, COCO and detector-specific views, and an unlabeled
`data/training.index.json` with matching image IDs. The views link to the
extracted images. Use stored image pixels (before EXIF display rotation) for
boxes, masks and crops.

## Detector training

Use SAM or another segmenter to make a COCO JSON with RLE masks. Match the
image, annotation and category IDs in `${gt}.json` (categories 1–34). Put
each mask in `segmentation` with `size: [height, width]` and `counts` as a
compressed ASCII string or an integer array. Include a donor from at least one
of the weighted Co-DETR and TinyFormer classes (zero-based IDs 23, 29, 30 and 31).

```bash
python -m training.prepare_sam prepare-masks \
  --masks inputs/masks.rle.json \
  --annotations "${gt}.json" --images "$images" --mask-output inputs/masks.json

python -m training.prepare_sam build-bank \
  --masks inputs/masks.json \
  --annotations "${gt}_0based.json" --images "$images" --output-root inputs/donors

python -m training.prepare_sam import \
  --bank inputs/donors --masks inputs/masks.json \
  --annotations "${gt}_0based.json" --images "$images" \
  --mask-output data/augmentation/instances_train_sam.json \
  --output-root data/augmentation/sam_object_bank --data-config data/data.yaml
```

The import updates `data/data.yaml` with the prepared inputs. Co-DETR and
TinyFormer use the RGBA bank; RF-DETR uses the RLE masks.

Obtain the initializers from the publishers and save them to the paths in the
[initializer catalog](../configs/upstream_initialization.json). Use TinyFormer's
Objects365+COCO XL-PBM weights and RF-DETR's legacy `rf-detr-large.pth`. For
DINOv3, obtain the native PyTorch backbones through
[Meta's access form](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/).
SigLIP 2 initializes through timm. The helper below reads the catalog paths.

```bash
initial() {
  python -c 'import json, sys; print(json.load(open("configs/upstream_initialization.json"))["artifacts"][sys.argv[1]]["path"])' "$1"
}

for model in tinyformer rfdetr yolo26; do
  python -m training.train_detector --model "$model" --data data/data.yaml \
    --initial "$(initial "${model}_initial")" --output-dir "outputs/$model"
done

python scripts/codetr_python.py -m torch.distributed.run --standalone \
  --nproc-per-node=2 --module training.train_codetr --data data/data.yaml \
  --initial "$(initial codetr_initial)" --output-dir outputs/codetr
```

## Crop-classifier training

Export the TinyFormer-XL-PBM EMA from your detector run, then generate proposals
on the prepared training images. This writes a COCO JSON array with canonical
`image_id`, `category_id` (1–34), pixel XYWH `bbox` and `score`.

```bash
cat > local-detector.json <<'EOF'
{"tinyformer": "outputs/tinyformer/checkpoint_terminal_ema.pth"}
EOF

python -m training.export_models --model tinyformer --sources local-detector.json \
  --directory outputs/proposal-models --trust-native

python -m inference.detectors --model tinyformer --index data/training.index.json \
  --images "$images" --checkpoint-root outputs/proposal-models \
  --manifest outputs/proposal-models/checkpoint-manifest.json \
  --output inputs/train-proposals.json
```

```bash
records=data/dcr/train_all.records.jsonl
python -m training.prepare_dcr_records generate --annotations "${gt}.json" \
  --proposals inputs/train-proposals.json --images "$images" --output "$records"
```

The generator writes a manifest next to the records. The sampler needs all 34
foreground classes and both hard and regular background crops. To use existing
records and their manifest, see `python -m training.prepare_dcr_records import --help`.

The recipe IDs `hplus`, `v7` and `siglip` select DINOv3 ViT-H+/16 with LoRA,
DINOv3 ViT-L/16 and SigLIP 2 ViT-L/16, respectively.

```bash
python -m training.train_dcr --model hplus --records "$records" \
  --allow-new-records --image-root "$images" \
  --initial-backbone "$(initial hplus_base)" \
  --output outputs/dinov3-vit-h16plus-lora.pt

python -m training.train_dcr --model v7 --records "$records" \
  --allow-new-records --image-root "$images" \
  --initial-backbone "$(initial v7_base)" --output outputs/dinov3-vit-l16.pt

python -m training.train_dcr --model siglip --records "$records" \
  --allow-new-records --image-root "$images" --output outputs/siglip2-vit-l16-384.pt
```

## Export and run

Create `local-models.json` in the repository root. Paths are relative to this
file. The H+ backbone is packaged with the trained adapters and readout.

```bash
cat > local-models.json <<EOF
{
  "codetr": "outputs/codetr/checkpoint_terminal_ema.pth",
  "tinyformer": "outputs/tinyformer/checkpoint_terminal_ema.pth",
  "rfdetr": "outputs/rfdetr/checkpoint_terminal_ema.pth",
  "yolo26": "outputs/yolo26/stage2/checkpoint_fixed_epoch_7.pt",
  "hplus": "outputs/dinov3-vit-h16plus-lora.pt",
  "v7": "outputs/dinov3-vit-l16.pt",
  "siglip": "outputs/siglip2-vit-l16-384.pt",
  "hplus_base": "$(initial hplus_base)"
}
EOF
```

Export using the native checkpoints you produced and trust:

```bash
python -m training.export_models --sources local-models.json \
  --directory outputs/local-models --trust-native
```

Run the [inference commands](../README.md#quick-start) with
`--checkpoint-root outputs/local-models` and
`--manifest outputs/local-models/checkpoint-manifest.json`.
