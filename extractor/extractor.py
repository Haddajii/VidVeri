import cv2
import os

def extract_frames(video_path, output_folder, frame_rate=1):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)

    count = 0        # counts how many frames we have processed
    frame_id = 0     # counts how many frames we saved
    fps = cap.get(cv2.CAP_PROP_FPS)  # frames per second of the video
    interval = int(fps * frame_rate) # save one frame every `frame_rate` seconds

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{frame_id}.jpg")
            cv2.imwrite(frame_filename, frame)
            print(f"Saved {frame_filename}")
            frame_id += 1

        count += 1

    cap.release()
    print("Done extracting frames")

if __name__ == "__main__":
    extract_frames(r"\Users\Mega-PC\Videos\Captures\React App - Google Chrome 2025-01-17 22-25-21.mp4", "output_frames", frame_rate=1)
