# 地块模型独立推理测试

默认读取项目根目录的 `result.tif` 和 `checkpoint/DelineateAnythingv2.pt`，
按约 0.25 米地面采样间距降采样，进行整幅影像实例分割。
脚本直接使用本地权重，不下载模型；不修改主应用。

在项目根目录运行：

```powershell
& D:\miniconda3\envs\farm_edge\python.exe roi_test\infer.py
```

比较其他分辨率，并保存到不同目录：

```powershell
& D:\miniconda3\envs\farm_edge\python.exe roi_test\infer.py --resolution 0.10 --imgsz 2048 --output roi_test\outputs_010m
```

主要参数：`--conf` 最低置信度（默认 0.15），`--iou` 检测去重阈值（默认 0.5），
`--min-area` 最小区域面积（默认 10 平方米），`--device cpu` 使用处理器。
模型输入长边由 `--imgsz` 控制：识别影像大于该值时，模型内部仍会缩小影像。
此脚本用于当前小范围影像的整幅测试，没有实现作者完整的多窗口融合流程。

输出到 `roi_test/outputs/`：

| 文件 | 内容 |
| --- | --- |
| `inference_rgba.tif` | 带源坐标参考系的降采样影像，第四波段是有效像元透明度 |
| `input_preview.jpg` | 推理输入预览 |
| `overlay.png` | 地块轮廓和编号叠加图 |
| `fields.geojson` | 世界大地坐标系经纬度多边形，含编号、置信度和平方米面积 |
| `field_ids.tif` | 带源坐标参考系的地块编号栅格，0 为背景 |
| `summary.json` | 参数、尺寸、采样间距、区域数量及运行耗时 |

掩膜会扣除无效像元；重叠区域优先归属于置信度更高的候选。
同一候选的多个不连通区域会分别输出，保留原候选编号便于核查。
结果边界来自降采样栅格，没有进行原图边界细化；区域编号并不保证对应真实独立田块。

环境依赖：`torch`、`ultralytics`、`rasterio`、`numpy`、`pillow`、`shapely`、`pyproj`。
当前项目的 `farm_edge` 环境已具备上述依赖。

模型来源：[作者仓库](https://github.com/Lavreniuk/Delineate-Anything)。
