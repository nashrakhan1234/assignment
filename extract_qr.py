import cv2

def extract_qr_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    detector = cv2.QRCodeDetector()
    decoded_texts = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect and decode QR
        data, points, _ = detector.detectAndDecode(frame)
        if data:
            decoded_texts.add(data)

    cap.release()
    return decoded_texts

if __name__ == "__main__":
    video_with_qr = "recorded_test.mp4"   # Path to QR-embedded video
    texts = extract_qr_from_video(video_with_qr)
    print("📥 Extracted texts:", texts)
