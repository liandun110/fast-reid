# 依次对图像序列执行检测行人、裁剪行人、提取reid特征三步操作。
SEQPATH=datasets/yisuo/人脸追踪02
python datasets/person_detect.py --seq_path=$SEQPATH
python datasets/crop_person.py --seq_path=$SEQPATH
python demo/demo.py --config-file configs/DukeMTMC/bagtricks_R101-ibn.yml --seq_path=$SEQPATH --opts MODEL.WEIGHTS datasets/duke_bot_R101-ibn.pth