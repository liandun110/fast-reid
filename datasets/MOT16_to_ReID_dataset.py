import os
import cv2
import argparse
import pandas as pd
import tqdm

def save_cropped_pedestrians(args):
    gt_path = os.path.join(args.seq_path, 'gt/gt.txt')
    gt_df = pd.read_csv(gt_path, header=None)
    gt_df.columns = ['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'class', 'vis']
    gt_df = gt_df[(gt_df['class'] == 1) & (gt_df['id'] > 0)]  # 只保留行人

    img_dir = os.path.join(args.seq_path, 'img1')
    frame_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])

    os.makedirs(args.output, exist_ok=True)

    for frame_idx, frame_file in tqdm.tqdm(enumerate(frame_files, 1), total=len(frame_files)):
        frame_path = os.path.join(img_dir, frame_file)
        img = cv2.imread(frame_path)
        if img is None:
            print(f"警告: 无法读取图片 {frame_path}")
            continue

        frame_gt = gt_df[gt_df['frame'] == frame_idx]
        frame_name_no_ext = os.path.splitext(frame_file)[0]

        for _, row in frame_gt.iterrows():
            pid = int(row['id'])
            x, y, w, h = map(int, [row['x'], row['y'], row['w'], row['h']])

            # 裁剪行人子图，防止越界
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(img.shape[1], x + w), min(img.shape[0], y + h)
            cropped = img[y1:y2, x1:x2]

            if cropped.size == 0:
                continue

            person_id_str = f"{pid:04d}"
            filename = f"{person_id_str}_c1s1_{frame_name_no_ext}_00.jpg"
            save_path = os.path.join(args.output, filename)
            cv2.imwrite(save_path, cropped)

def get_parser():
    parser = argparse.ArgumentParser(description="Crop pedestrians from MOT16 img1 frames and save as JPG.")
    parser.add_argument("--seq_path", default="datasets/MOT16/train/MOT16-04", help="Path to MOT16 sequence")
    parser.add_argument("--output", default="datasets/MOT16-04-ReID", help="Directory to save cropped images")
    return parser

if __name__ == "__main__":
    args = get_parser().parse_args()
    save_cropped_pedestrians(args)
