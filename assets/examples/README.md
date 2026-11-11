# Example prediction key

The overlays show up to 12 final predictions per NAMR training image at
confidence ≥ 0.20.

## Shoreline debris

[Full-size numbered image](predictions-coast.webp)

| Box | Class | Confidence |
| ---: | :--- | ---: |
| 1 | Plastic bottle cap | 0.82 |
| 2 | Disposable food container | 0.74 |
| 3 | Net-like item | 0.55 |

## Dense debris

[Full-size numbered image](predictions-dense.webp)

| Class | Box number (confidence) |
| :--- | :--- |
| Lighter | 1 (0.86) |
| Soft float | 2 (0.74), 6 (0.61), 9 (0.57), 11 (0.56) |
| Net-like item | 3 (0.69), 4 (0.64) |
| Anthropogenic fragment | 5 (0.62), 7 (0.58), 8 (0.58), 10 (0.57), 12 (0.56) |

The National Academy of Marine Research (NAMR / 國家海洋研究院) published
`2024_TWN_CWH_336.jpg` (shoreline) and `2025_TWN_TPH_41.jpg` (dense) in
[MDImageNet v1.0](https://mdimagenet.namr.gov.tw/namrOceanTrash/dataset/detail/JQdYLLKk9Qelnk6xAkqeDQ%3D%3D).
We resized the images, added predictions and trimmed display margins and
legends. The source images use [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
