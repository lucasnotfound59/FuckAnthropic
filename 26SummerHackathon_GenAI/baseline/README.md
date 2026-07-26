- 训练（含环境配置）
```shell
python train.py --epochs 30 --device cuda:0
```

- 验证
```shell
python val.py --weights results/exp/weights/best.pt --device cuda:0
```

- 将输出结果转换为一个csv，作为submission
```shell
python test.py --weights results/exp/weights/best.pt --device cuda:0 --save-csv
```