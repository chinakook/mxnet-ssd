# 1.准备数据
* DATA_DIR: 标注数据文件夹(包含".jpg"原始图像文件 和 ".xml"标注文件) ; 
* 运行后, .rec文件和.lst文件会存储于data文件夹下
* 程序结束会显示所有文件数(total file)

* run
```
python tools/prepare_db.py --root /mnt/data/tmp/flower/Archive20170906/1605 --target ./data/train.lst
python tools/prepare_db.py --root E:\DDL\dg_data\arrow\re2012 --target ./data/train.lst
```
# 2.开始训练
* --train-path: .rec文件路径
* --train-list: .lst文件路径
* --network: 基础网络，mobilenet_little是为dg定制的，mobilenet是通用的，前者不行可尝试后者
* --batch-size: batch size
* --pretrained: 预训练模型路径(含模型前缀)
* --epoch: 预训练模型epoch数
* --gpus: 选择第几个显卡 或者 0,1,2....(如果有多个)
* --num-example: 训练图像的总数
* --end-epoch: 训练迭代次数
* --data-shape: 模型输入size
* --num-class: 总类数 （为1）
* --num-example: 训练图像的总数 
* --class-name: 类别名字 （" "）

* run
```
python train.py  --train-path "./data/train.rec" --train-list "./data/train.lst" --val-path "" --gpus 0  --batch-size 16 --data-shape 608 --num-example 194 --num-class 1 --class-names "obj" --network "mobilenet_little" --pretrained "./model/pre/mobilenet" --freeze "" --lr 0.002 --end-epoch 60 --wd 0.0001
```
 python train.py --train-path ./data/train.rec --train-list ./data/train.lst --val-path "" --network "mobilenet_little" --batch-size 4 --pretrained "./model/pre/mobilenet" --epoch 1 --prefix ./model/ssd_mobilenet_little --gpus 0 --end-epoch 200 --data-shape 300 --num-class 1 --num-example 630 --class-name "obj"
# 3.测试模型
* python test.py 模型路径(含前缀) 模型迭代次数 待测试图像路径 测试结果保存路径

* run
```
python test.py ./model/ssd_mobilenet_300 60 /mnt/data/tmp/flower/Archive20170906/1605 ./test
python test.py ./model/ssd_mobilenet_little_300 60 e:\1 e:\1\1
```
# 4.deploy
* python deploy.py --network 网络模型 --epoch 迭代次数 --prefix 网络路径（含前缀） --data-shape 300  --num-class 1
* 输出 deploy_ssd_mobilenet_little 模型
* run
'''
python deploy.py --network mobilenet_little --epoch 60 --prefix ./model/ssd_mobilenet_little_608 --data-shape 608 --num-class 1
'''
python dg_data_imrec.py
python mbn_gluon.py