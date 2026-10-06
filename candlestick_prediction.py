import argparse
import time

import albumentations as A
import cv2
import numpy as np
import torch
from colorama import Fore
from mss import mss
from PIL import Image

from model import ViT

CLASSES = ['doji', 'bullish_engulfing', 'bearish_engulfing', 'morning_star', 'evening_star']
BAR_COLORS = [
    (156, 220, 235), (166, 207, 140), (236, 171, 193), (202, 163, 232), (255, 128, 128)
]
WINDOW_NAME = 'Candlestick Predictor'


def render_preview(probs, prediction, width=420, row_h=44):
    """Small standalone panel (no screen pixels), so it can't feed back into the capture."""
    height = row_h * (len(CLASSES) + 1)
    panel = np.full((height, width, 3), 40, dtype=np.uint8)
    label = f'{CLASSES[prediction]} - {probs[prediction]:.3f}'
    cv2.rectangle(panel, (0, 0), (width, row_h), BAR_COLORS[prediction], -1)
    cv2.putText(panel, label, (10, row_h - 14), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    for i, name in enumerate(CLASSES):
        y0 = row_h * (i + 1)
        cv2.rectangle(panel, (0, y0 + 4), (int(width * probs[i]), y0 + row_h - 4), BAR_COLORS[i], -1)
        cv2.putText(panel, f'{name} - {probs[i]:.3f}', (10, y0 + row_h - 14),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return panel


def render_full_overlay(raw_image, probs, prediction, scale_x, scale_y):
    """Original 4K-style overlay drawn on top of the captured screen (for screen recordings)."""
    render_image = cv2.cvtColor(np.array(raw_image), cv2.COLOR_RGB2BGR)

    overlay = render_image.copy()
    cv2.rectangle(
        overlay,
        (int(1350 * scale_x), int(590 * scale_y)),
        (int(2224 * scale_x), int(1076 * scale_y)),
        (128, 128, 128),
        -1
    )
    render_image = cv2.addWeighted(overlay, 0.7, render_image, 1 - 0.7, 0)

    for x, class_name in enumerate(CLASSES):
        label = f'{class_name} - {round(probs[x], 3)}'
        y_min = int(((x + 1) * 100 + 490) * scale_y)
        y_max = int(y_min + 100 * scale_y)
        x_max = int((1350 + 800 * probs[x]) * scale_x)
        render_image = cv2.rectangle(render_image, (int(1350 * scale_x), y_min), (x_max, y_max), BAR_COLORS[x], -1)
        render_image = cv2.putText(
            render_image, label, (int(1350 * scale_x), int(y_min + 75 * scale_y)),
            cv2.FONT_HERSHEY_DUPLEX, 1.5 * scale_x, (0, 0, 0), max(1, int(3 * scale_x)), cv2.LINE_AA
        )

    render_image = cv2.rectangle(
        render_image,
        (int(2224 * scale_x), int(590 * scale_y)),
        (int(3420 * scale_x), int(1455 * scale_y)),
        BAR_COLORS[prediction],
        max(1, int(10 * scale_x))
    )
    render_image = cv2.rectangle(
        render_image,
        (int(1350 * scale_x), int(400 * scale_y)),
        (int(3420 * scale_x), int(590 * scale_y)),
        BAR_COLORS[prediction],
        -1
    )
    label = f'{CLASSES[prediction]} - {round(probs[prediction], 3)}'
    return cv2.putText(
        render_image, label, (int(1350 * scale_x), int(550 * scale_y)),
        cv2.FONT_HERSHEY_DUPLEX, 3 * scale_x, (0, 0, 0), max(1, int(8 * scale_x)), cv2.LINE_AA
    )


def main():
    parser = argparse.ArgumentParser(description='Live candlestick pattern predictor')
    parser.add_argument('--monitor', type=int, default=1, help='Monitor to capture (1 = primary)')
    parser.add_argument('--fps', type=float, default=5, help='Maximum frames per second')
    parser.add_argument('--full-overlay', action='store_true',
                        help='Draw the full-screen overlay on the captured frame (for screen recordings)')
    args = parser.parse_args()

    # Keep CPU inference from saturating every core
    torch.set_num_threads(2)

    model = ViT()
    model.load_state_dict(torch.load('checkpoints/25_model.pt', weights_only=True, map_location=torch.device('cpu')))
    model.eval()

    frame_time = 1.0 / max(args.fps, 0.1)

    try:
        with mss() as sct:
            if not 1 <= args.monitor < len(sct.monitors):
                print(f'Invalid --monitor {args.monitor}. Available: 1-{len(sct.monitors) - 1}')
                return

            monitor = sct.monitors[args.monitor]
            screen_w = monitor['width']
            screen_h = monitor['height']

            # Scale relative to 4K (3840x2160), which the original model data was captured on
            scale_x = screen_w / 3840
            scale_y = screen_h / 2160
            y_top = int(170 * scale_y)

            transforms = A.Compose([
                A.Crop(x_min=0, y_min=y_top, x_max=screen_w, y_max=screen_h),
                A.Resize(700, 500),
                A.Resize(224, 224),
                A.Crop(x_min=130, y_min=43, x_max=202, y_max=147),  # eight candles
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                A.ToTensorV2(),
            ])

            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            if not args.full_overlay:
                # Park the small panel bottom-left, away from the chart area the model reads
                cv2.resizeWindow(WINDOW_NAME, 420, 264)
                cv2.moveWindow(WINDOW_NAME, monitor['left'] + 10, monitor['top'] + screen_h - 340)
            else:
                cv2.resizeWindow(WINDOW_NAME, 960, 540)

            print(f"Running on monitor {args.monitor} ({screen_w}x{screen_h}). Press 'q' or Esc in the window, or Ctrl+C, to stop.")

            with torch.inference_mode():
                while True:
                    start = time.monotonic()

                    sct_image = sct.grab(monitor)
                    raw_image = Image.frombytes('RGB', sct_image.size, sct_image.rgb)
                    img = transforms(image=np.array(raw_image)[:, :, :3])['image']

                    probs = torch.softmax(model(img.unsqueeze(0)), dim=1)[0]
                    prediction = int(torch.argmax(probs))
                    probs = probs.tolist()
                    print(Fore.LIGHTYELLOW_EX + f'{prediction} {probs[prediction]}' + Fore.RESET)

                    if args.full_overlay:
                        frame = render_full_overlay(raw_image, probs, prediction, scale_x, scale_y)
                    else:
                        frame = render_preview(probs, prediction)
                    cv2.imshow(WINDOW_NAME, frame)

                    # Stop on q / Esc, or if the window was closed
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord('q'), 27) or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                        break

                    time.sleep(max(0.0, frame_time - (time.monotonic() - start)))
    except KeyboardInterrupt:
        print('\nInterrupted by user')
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
