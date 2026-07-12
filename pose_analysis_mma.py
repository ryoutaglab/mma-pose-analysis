import sys
import os
import cv2
import numpy as np
from datetime import datetime
from ultralytics import YOLO

# キーポイントインデックス定数（YOLOv8 17点モデル）
LEFT_EAR       = 3
RIGHT_EAR      = 4
LEFT_SHOULDER  = 5
RIGHT_SHOULDER = 6
LEFT_HIP       = 11
RIGHT_HIP      = 12

# 解析する最大人数
MAX_PERSONS = 2

# スケルトン接続定義（0-indexed）
SKELETON = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
    [5, 11],  [6, 12],  [5, 6],   [5, 7],   [6, 8],
    [7, 9],   [8, 10],  [1, 2],   [0, 1],   [0, 2],
    [1, 3],   [2, 4],
]

# 人物ごとの描画色（BGR）
PERSON_COLORS = [
    (0, 255, 255),   # 黄
    (0, 165, 255),   # オレンジ
    (255, 0, 255),   # マゼンタ
]

# 信頼度しきい値
CONF_THRESH = 0.5


def resolve_source(argv):
    if len(argv) > 1:
        return int(argv[1]) if argv[1].isdigit() else argv[1]
    return 'sample.mp4'


def resolve_max_persons(argv):
    if len(argv) > 2:
        try:
            return max(1, int(argv[2]))
        except ValueError:
            pass
    return MAX_PERSONS


def make_output_path(source):
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    if isinstance(source, int):
        return f'output_{timestamp}.mp4'
    base, _ = os.path.splitext(source)
    return f'{base}_{timestamp}.mp4'


def draw_person(frame, kp, conf, color):
    for i, (x, y) in enumerate(kp):
        if x > 0 and y > 0 and conf[i] > CONF_THRESH:
            cv2.circle(frame, (int(x), int(y)), 4, color, cv2.FILLED)

    for a, b in SKELETON:
        if a >= len(kp) or b >= len(kp):
            continue
        x1, y1 = kp[a]
        x2, y2 = kp[b]
        if (x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0
                and conf[a] > CONF_THRESH and conf[b] > CONF_THRESH):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)


def draw_midline(frame, kp, conf):
    if len(kp) <= RIGHT_HIP:
        return
    pts = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]
    if any(conf[i] <= CONF_THRESH or kp[i][0] == 0 for i in pts):
        return

    mid_sh  = (int((kp[LEFT_SHOULDER][0] + kp[RIGHT_SHOULDER][0]) / 2),
               int((kp[LEFT_SHOULDER][1] + kp[RIGHT_SHOULDER][1]) / 2))
    mid_hip = (int((kp[LEFT_HIP][0] + kp[RIGHT_HIP][0]) / 2),
               int((kp[LEFT_HIP][1] + kp[RIGHT_HIP][1]) / 2))

    cv2.line(frame, mid_sh, mid_hip, (0, 255, 255), 4)
    cv2.circle(frame, mid_sh,  5, (0, 0, 255), cv2.FILLED)
    cv2.circle(frame, mid_hip, 5, (0, 0, 255), cv2.FILLED)

    # 肩中点から耳への線（両耳→中点、片耳→その耳、両方なし→スキップ）
    if len(kp) > RIGHT_EAR:
        l_ok = conf[LEFT_EAR]  > CONF_THRESH and kp[LEFT_EAR][0]  > 0
        r_ok = conf[RIGHT_EAR] > CONF_THRESH and kp[RIGHT_EAR][0] > 0
        if l_ok and r_ok:
            head_pt = (int((kp[LEFT_EAR][0] + kp[RIGHT_EAR][0]) / 2),
                       int((kp[LEFT_EAR][1] + kp[RIGHT_EAR][1]) / 2))
        elif l_ok:
            head_pt = (int(kp[LEFT_EAR][0]),  int(kp[LEFT_EAR][1]))
        elif r_ok:
            head_pt = (int(kp[RIGHT_EAR][0]), int(kp[RIGHT_EAR][1]))
        else:
            head_pt = None
        if head_pt is not None:
            cv2.line(frame, mid_sh, head_pt, (0, 255, 255), 4)
            cv2.circle(frame, head_pt, 5, (0, 0, 255), cv2.FILLED)


def select_top_persons(r, max_n):
    """面積の大きい上位 max_n 人のインデックスを返す"""
    if r.boxes is None or len(r.boxes) == 0:
        return []
    boxes = r.boxes.xyxy.cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return np.argsort(areas)[::-1][:max_n].tolist()


def main():
    source      = resolve_source(sys.argv)
    max_persons = resolve_max_persons(sys.argv)
    out_path    = make_output_path(source)

    model = YOLO('yolo26l-pose.pt')
    cap   = cv2.VideoCapture(source)

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    print(f'解析開始: {source}  →  出力: {out_path}')
    print("終了するには 'q' を押してください。")

    frame_count   = 0
    display_frame = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % 2 == 1 or display_frame is None:
            results = model(frame, stream=True, verbose=False, imgsz=640)

            canvas = frame.copy()

            for r in results:
                if r.keypoints is None:
                    continue

                top_idx  = select_top_persons(r, max_persons)
                kp_all   = r.keypoints.xy.cpu().numpy()
                conf_all = (r.keypoints.conf.cpu().numpy()
                            if r.keypoints.conf is not None
                            else np.ones((len(kp_all), 17)))

                for rank, idx in enumerate(top_idx):
                    if idx >= len(kp_all):
                        continue
                    color = PERSON_COLORS[rank % len(PERSON_COLORS)]
                    kp    = kp_all[idx]
                    conf  = conf_all[idx]
                    draw_person(canvas, kp, conf, color)
                    draw_midline(canvas, kp, conf)

            display_frame = canvas

        out_frame = display_frame if display_frame is not None else frame
        writer.write(out_frame)
        cv2.imshow('MMA Pose Analysis', out_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f'完了: {out_path}')


if __name__ == '__main__':
    main()
