import cv2
import qrcode
import numpy as np

def generate_qr_code(text, qr_size=300):
    """
    Generate a high-contrast QR code image.
    """
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=4,
    )
    qr.add_data(text)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    img = img.resize((qr_size, qr_size))
    return np.array(img)

def embed_qr_in_video(input_video, output_video, text):
    # Generate QR code
    qr_img = generate_qr_code(text, qr_size=300)

    # Open input video
    cap = cv2.VideoCapture(input_video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Use MP4 codec for Windows compatibility
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Center the QR code
        x_offset = width // 2 - qr_img.shape[1] // 2
        y_offset = height // 2 - qr_img.shape[0] // 2
        y1, y2 = y_offset, y_offset + qr_img.shape[0]
        x1, x2 = x_offset, x_offset + qr_img.shape[1]

        # Overlay QR code
        frame[y1:y2, x1:x2] = qr_img

        out.write(frame)

    cap.release()
    out.release()
    print(f"✅ Video saved with embedded QR at {output_video}")

if __name__ == "__main__":
    input_video = "recorded_test.mp4"                 # Input video path
    output_video = "output_qr_centered.mp4"   # Output video path
    secret_text = "Hello Mission 65"

    embed_qr_in_video(input_video, output_video, secret_text)
