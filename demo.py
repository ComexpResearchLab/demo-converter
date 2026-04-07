#!/usr/bin/env python3

import subprocess
import json
import os
import select
import time
from threading import Thread

import numpy as np
from PIL import Image
from collections import deque
from tqdm import tqdm


def get_video_info(path):
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,r_frame_rate,nb_frames',
        '-of', 'json',
        path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)['streams'][0]
    w, h = int(info['width']), int(info['height'])
    num, den = map(int, info['r_frame_rate'].split('/'))
    fps = num / den if den else 0
    nb_frames = int(info['nb_frames']) if info.get('nb_frames', 'N/A') != 'N/A' else None
    return w, h, fps, nb_frames


def cropdetect(frame, threshold=24):
    row_means = frame.mean(axis=1)
    col_means = frame.mean(axis=0)

    rows_above = np.where(row_means > threshold)[0]
    cols_above = np.where(col_means > threshold)[0]

    if len(rows_above) == 0 or len(cols_above) == 0:
        return 0, 0, frame.shape[1] - 1, frame.shape[0] - 1

    return int(cols_above[0]), int(rows_above[0]), int(cols_above[-1]), int(rows_above[-1])


class CropEstimator:
    def __init__(self, window=120):
        self.crops = deque(maxlen=window)

    def update(self, crop):
        self.crops.append(crop)

    def get(self):
        arr = np.array(self.crops)
        return tuple(int(v) for v in np.median(arr, axis=0))


def process(input_path, output_dir, crop_window=120, threshold=24,
            stall_timeout_s=30.0):
    w, h, fps, nb_frames = get_video_info(input_path)
    frame_size = w * h

    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        'ffmpeg', '-v', 'error',
        '-i', input_path,
        '-pix_fmt', 'gray',
        '-f', 'rawvideo',
        'pipe:1'
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    stderr_lines = []
    def drain_stderr():
        for line in proc.stderr:
            stderr_lines.append(line.decode('utf-8', errors='ignore').rstrip())
    stderr_thread = Thread(target=drain_stderr, daemon=True)
    stderr_thread.start()

    estimator = CropEstimator(window=crop_window)

    written = 0
    frame_nb = 0
    buf = bytearray()
    stdout_fd = proc.stdout.fileno()
    chunk_size = min(frame_size * 8, 1024 * 1024)
    last_progress = time.monotonic()

    pbar = tqdm(total=nb_frames, unit='fr', desc='Decoding',
                dynamic_ncols=True, miniters=1)
    try:
        while True:
            ready, _, _ = select.select([stdout_fd], [], [], 1.0)
            if ready:
                data = os.read(stdout_fd, chunk_size)
                if not data:
                    break
                buf.extend(data)
                last_progress = time.monotonic()
            else:
                if proc.poll() is not None:
                    break
                if time.monotonic() - last_progress > stall_timeout_s:
                    proc.kill()
                    raise TimeoutError(
                        f'ffmpeg stalled: no output for {stall_timeout_s:.1f}s '
                        f'while processing {input_path}')

            while len(buf) >= frame_size:
                frame = np.frombuffer(buf[:frame_size], dtype=np.uint8).reshape(h, w)
                del buf[:frame_size]

                crop = cropdetect(frame, threshold)
                estimator.update(crop)
                x1, y1, x2, y2 = estimator.get()

                cropped = frame[y1:y2 + 1, x1:x2 + 1]
                if cropped.size == 0:
                    frame_nb += 1
                    pbar.update(1)
                    continue

                thumb = np.array(
                    Image.fromarray(cropped).resize((16, 16), Image.BILINEAR)
                )

                # -- NOTE: В продакшн именно здесь thumb фреймы бы просто уходили в нашу сторону --
                if frame_nb % 10 == 0:
                    Image.fromarray(thumb).save(
                        os.path.join(output_dir, f'frame_{frame_nb:06d}.png')
                    )
                    written += 1

                frame_nb += 1
                pbar.update(1)
    finally:
        pbar.close()
        proc.stdout.close()
        proc.wait()
        stderr_thread.join(timeout=5)

    if proc.returncode != 0:
        err = '\n'.join(stderr_lines[-20:]) if stderr_lines else '(no output)'
        print(f'WARNING: ffmpeg exited with code {proc.returncode}:\n{err}')

    print(f'Готово: {frame_nb} фреймов декодировано, {written} кадров в {output_dir}/')
    if estimator.crops:
        x1, y1, x2, y2 = estimator.get()


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(
        description='Демо сбора данных для TAPe преобразования'
    )
    p.add_argument('input', help='Путь к видео')
    p.add_argument('-o', '--output', default='demo_frames',
                   help='Выходная папка для фреймов (для дебага демо, чтобы визуально увидеть результат)')
    args = p.parse_args()

    process(args.input, args.output)
