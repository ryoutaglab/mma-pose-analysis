import sys
import os
import argparse
import tempfile
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

# スケルトン接続定義（0-indexed）
SKELETON = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
    [5, 11],  [6, 12],  [5, 6],   [5, 7],   [6, 8],
    [7, 9],   [8, 10],  [1, 2],   [0, 1],   [0, 2],
    [1, 3],   [2, 4],
]

# 人物ごとの描画色（BGR） rank0=P1, rank1=P2
PERSON_COLORS = [
    (0, 255, 255),   # 黄（P1）
    (0, 165, 255),   # オレンジ（P2）
]

# 信頼度しきい値
CONF_THRESH = 0.5

# ID消失時に別IDへ再割当を許容する距離しきい値（画面幅に対する比率）
LOST_MATCH_RATIO = 0.30


def make_output_path(source):
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    if isinstance(source, int):
        return f'output_{timestamp}.mp4'
    base, _ = os.path.splitext(source)
    return f'{base}_{timestamp}.mp4'


def make_tracker_config(track_buffer_frames):
    """track_bufferを反映したByteTrack設定を一時ファイルに書き出す"""
    content = (
        "tracker_type: bytetrack\n"
        "track_high_thresh: 0.25\n"
        "track_low_thresh: 0.1\n"
        "new_track_thresh: 0.25\n"
        f"track_buffer: {track_buffer_frames}\n"
        "match_thresh: 0.8\n"
        "fuse_score: True\n"
    )
    fd, path = tempfile.mkstemp(suffix='_bytetrack.yaml')
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        f.write(content)
    return path


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


def get_boxes_and_ids(r):
    """検出結果からbox配列とtracker_id配列を取得（IDが無ければNone）"""
    if r is None or r.boxes is None or len(r.boxes) == 0:
        return None, None
    boxes = r.boxes.xyxy.cpu().numpy()
    tracker_ids = r.boxes.id.int().cpu().numpy() if r.boxes.id is not None else None
    return boxes, tracker_ids


def auto_assign_ids(boxes, tracker_ids):
    """2人以上検出された最初のフレームで、左右位置からP1・P2のIDを決定する"""
    if tracker_ids is None or len(tracker_ids) < 2:
        return None, None

    centers_x = [(boxes[i][0] + boxes[i][2]) / 2 for i in range(len(boxes))]
    areas = [(boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])
             for i in range(len(boxes))]
    top2_idx = np.argsort(areas)[::-1][:2]

    left_idx  = min(top2_idx, key=lambda i: centers_x[i])
    right_idx = max(top2_idx, key=lambda i: centers_x[i])

    p1_id = int(tracker_ids[left_idx])
    p2_id = int(tracker_ids[right_idx])
    return p1_id, p2_id


def select_by_id(boxes, tracker_ids, p1_id, p2_id):
    """tracker_idでP1・P2に対応するインデックスを選択する"""
    persons = {}
    if tracker_ids is None:
        return persons
    for i, tid in enumerate(tracker_ids):
        if int(tid) == p1_id:
            persons[0] = i  # rank=0がP1
        elif int(tid) == p2_id:
            persons[1] = i  # rank=1がP2
    return persons


def find_lost_person(boxes, tracker_ids, last_center, assigned_ids, frame_width):
    """IDが消えた時に、最後の座標から最も近い未割当の人物を探す"""
    if last_center is None or tracker_ids is None:
        return None

    threshold = frame_width * LOST_MATCH_RATIO
    best_idx  = None
    best_dist = float('inf')

    for i, tid in enumerate(tracker_ids):
        if int(tid) in assigned_ids:
            continue  # 既に割り当て済みのIDはスキップ

        center_x = (boxes[i][0] + boxes[i][2]) / 2
        center_y = (boxes[i][1] + boxes[i][3]) / 2

        dist = np.sqrt(
            (center_x - last_center[0]) ** 2 +
            (center_y - last_center[1]) ** 2
        )

        if dist < threshold and dist < best_dist:
            best_dist = dist
            best_idx  = i

    return best_idx


def resolve_persons(boxes, tracker_ids, p1_id, p2_id, last_centers, frame_width):
    """
    毎フレーム、P1・P2に対応するインデックスを解決する。
    IDがそのまま見つかればそれを使い、消えていれば距離ベースで復活を試みる。
    復活した場合はp1_id・p2_idを新しいIDに更新し、以降の追跡を継続する。
    """
    persons = select_by_id(boxes, tracker_ids, p1_id, p2_id)

    assigned_ids = set()
    if tracker_ids is not None:
        for rank in (0, 1):
            if rank in persons:
                assigned_ids.add(int(tracker_ids[persons[rank]]))

    for rank in (0, 1):
        if rank not in persons:
            idx = find_lost_person(boxes, tracker_ids, last_centers[rank],
                                    assigned_ids, frame_width)
            if idx is not None:
                persons[rank] = idx
                new_id = int(tracker_ids[idx])
                assigned_ids.add(new_id)
                if rank == 0:
                    p1_id = new_id
                else:
                    p2_id = new_id

    return persons, p1_id, p2_id


def manual_select_ids(frame, boxes, tracker_ids):
    """最初のフレームを表示し、クリックでP1・P2のIDを指定する"""
    display_frame = frame.copy()
    centers = []
    if boxes is not None and tracker_ids is not None:
        for i, tid in enumerate(tracker_ids):
            box = boxes[i]
            cx = int((box[0] + box[2]) / 2)
            cy = int((box[1] + box[3]) / 2)
            centers.append((cx, cy, int(tid)))
            cv2.putText(display_frame, f'ID:{int(tid)}', (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

    cv2.putText(display_frame,
                'Click P1 first, then P2. Enter to confirm, Esc to redo.',
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    click_points = []

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click_points.append((x, y))

    cv2.namedWindow('Select P1 and P2')
    cv2.setMouseCallback('Select P1 and P2', on_click)

    while True:
        frame_to_show = display_frame.copy()
        for i, (px, py) in enumerate(click_points[:2]):
            cv2.circle(frame_to_show, (px, py), 8, (0, 0, 255), cv2.FILLED)
            cv2.putText(frame_to_show, f'P{i + 1}', (px + 10, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow('Select P1 and P2', frame_to_show)

        key = cv2.waitKey(1) & 0xFF
        if key == 13 and len(click_points) >= 2:  # Enter
            break
        if key == 27:  # Esc: クリックをやり直す
            click_points.clear()
        if key == ord('q'):
            cv2.destroyWindow('Select P1 and P2')
            print('手動指定がキャンセルされました。')
            sys.exit(1)

    cv2.destroyWindow('Select P1 and P2')

    def nearest_id(point):
        if not centers:
            return None
        px, py = point
        return min(centers, key=lambda c: (c[0] - px) ** 2 + (c[1] - py) ** 2)[2]

    p1_id = nearest_id(click_points[0])
    p2_id = nearest_id(click_points[1])

    if p1_id is None or p2_id is None or p1_id == p2_id:
        print('P1・P2の指定に失敗しました。検出されたIDをクリックしてください。')
        sys.exit(1)

    print(f'P1 ID:{p1_id}（手動指定）/ P2 ID:{p2_id}（手動指定）')
    return p1_id, p2_id


def render_persons(canvas, r, boxes, tracker_ids, p1_id, p2_id, last_centers, frame_width):
    """P1・P2を解決して描画し、更新後のp1_id・p2_idを返す"""
    persons, p1_id, p2_id = resolve_persons(
        boxes, tracker_ids, p1_id, p2_id, last_centers, frame_width)

    kp_all   = r.keypoints.xy.cpu().numpy()
    conf_all = (r.keypoints.conf.cpu().numpy()
                if r.keypoints.conf is not None
                else np.ones((len(kp_all), 17)))

    for rank, idx in persons.items():
        if idx >= len(kp_all):
            continue
        color = PERSON_COLORS[rank % len(PERSON_COLORS)]
        draw_person(canvas, kp_all[idx], conf_all[idx], color)
        draw_midline(canvas, kp_all[idx], conf_all[idx])

        box = boxes[idx]
        last_centers[rank] = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

    return p1_id, p2_id


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', nargs='?', default='sample.mp4')
    parser.add_argument('--track', '-T', action='store_true',
                         help='最初のフレームでP1・P2を手動指定する')
    parser.add_argument('--track-buffer', type=float, default=3.0,
                         help='ID消失後の補完時間（秒）デフォルト3秒')
    return parser.parse_args()


def main():
    args   = parse_args()
    source = int(args.source) if str(args.source).isdigit() else args.source
    out_path = make_output_path(source)

    model = YOLO('yolo26x-pose.pt')
    cap   = cv2.VideoCapture(source)

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tracker_path = make_tracker_config(max(1, round(args.track_buffer * fps)))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    print(f'解析開始: {source}  →  出力: {out_path}')
    print("終了するには 'q' を押してください。")

    p1_id        = None
    p2_id        = None
    id_assigned  = False
    last_centers = {0: None, 1: None}
    frame_count   = 0
    display_frame = None

    if args.track:
        ret, frame = cap.read()
        if not ret:
            print('動画からフレームを取得できませんでした。')
            cap.release()
            writer.release()
            return
        frame_count = 1

        results = model.track(frame, persist=True, verbose=False,
                               imgsz=640, tracker=tracker_path)
        r = next(iter(results), None)
        boxes, tracker_ids = get_boxes_and_ids(r)

        p1_id, p2_id = manual_select_ids(frame, boxes, tracker_ids)
        id_assigned = True

        canvas = frame.copy()
        if r is not None and r.keypoints is not None and boxes is not None:
            p1_id, p2_id = render_persons(
                canvas, r, boxes, tracker_ids, p1_id, p2_id, last_centers, width)
        display_frame = canvas

        writer.write(display_frame)
        cv2.imshow('MMA Pose Analysis', display_frame)
        cv2.waitKey(1)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % 2 == 1 or display_frame is None:
            results = model.track(frame, persist=True, verbose=False,
                                   imgsz=640, tracker=tracker_path)

            canvas = frame.copy()

            for r in results:
                if r.keypoints is None:
                    continue

                boxes, tracker_ids = get_boxes_and_ids(r)
                if boxes is None:
                    continue

                if not id_assigned:
                    auto_p1, auto_p2 = auto_assign_ids(boxes, tracker_ids)
                    if auto_p1 is not None:
                        p1_id, p2_id = auto_p1, auto_p2
                        id_assigned = True
                        print(f'P1 ID:{p1_id}（左）/ P2 ID:{p2_id}（右）')

                if not id_assigned:
                    continue

                p1_id, p2_id = render_persons(
                    canvas, r, boxes, tracker_ids, p1_id, p2_id, last_centers, width)

            display_frame = canvas

        out_frame = display_frame if display_frame is not None else frame
        writer.write(out_frame)
        cv2.imshow('MMA Pose Analysis', out_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    if os.path.exists(tracker_path):
        os.remove(tracker_path)
    print(f'完了: {out_path}')


if __name__ == '__main__':
    main()
