#!/usr/bin/env python3
"""
Video DCT Steganography - Text Embedding Script (Improved with Lossless Options)
Hides text messages in video files using DCT coefficients.
"""

import cv2
import numpy as np
from scipy.fftpack import dct, idct
import struct
import sys
import os
from typing import Tuple

class VideoTextEmbedder:
    def __init__(self, block_size: int = 8, strength: float = 50.0):  # Increased default strength
        self.block_size = block_size
        self.strength = strength
        self.magic_header = b'DCTSTEG'

    def _apply_dct_2d(self, block: np.ndarray) -> np.ndarray:
        return dct(dct(block.T, norm='ortho').T, norm='ortho')

    def _apply_idct_2d(self, block: np.ndarray) -> np.ndarray:
        return idct(idct(block.T, norm='ortho').T, norm='ortho')

    def _text_to_bits(self, text: str) -> str:
        text_bytes = text.encode('utf-8')
        length = len(text_bytes)
        data = self.magic_header + struct.pack('<I', length) + text_bytes
        return ''.join(format(byte, '08b') for byte in data)

    def _embed_bit_in_dct_block(self, dct_block: np.ndarray, bit: int, pos: Tuple[int, int] = (1, 2)) -> np.ndarray:
        modified_block = dct_block.copy()
        coeff = modified_block[pos[0], pos[1]]
        
        # Stronger embedding with better separation
        if bit == 1:
            # Make coefficient clearly positive and large
            modified_block[pos[0], pos[1]] = abs(coeff) + self.strength
        else:
            # Make coefficient clearly positive but small, or negative
            if abs(coeff) > self.strength * 0.8:
                modified_block[pos[0], pos[1]] = abs(coeff) - self.strength * 0.8
            else:
                modified_block[pos[0], pos[1]] = -abs(coeff) - self.strength * 0.5
        
        return modified_block

    def _process_frame_for_embedding(self, frame: np.ndarray, bits: str, bit_index: int) -> Tuple[np.ndarray, int]:
        height, width = frame.shape
        result_frame = frame.copy()
        for i in range(0, height - self.block_size + 1, self.block_size):
            for j in range(0, width - self.block_size + 1, self.block_size):
                if bit_index >= len(bits):
                    return result_frame, bit_index
                block = frame[i:i+self.block_size, j:j+self.block_size].astype(np.float32)
                dct_block = self._apply_dct_2d(block)
                bit = int(bits[bit_index])
                modified_dct_block = self._embed_bit_in_dct_block(dct_block, bit)
                modified_block = self._apply_idct_2d(modified_dct_block)
                result_frame[i:i+self.block_size, j:j+self.block_size] = np.clip(modified_block, 0, 255).astype(np.uint8)
                bit_index += 1
        return result_frame, bit_index

    def embed_text_in_video(self, input_video_path: str, output_video_path: str, text: str, use_lossless: bool = True) -> bool:
        try:
            print(f"Opening video: {input_video_path}")
            cap = cv2.VideoCapture(input_video_path)
            if not cap.isOpened():
                print("Error: Could not open video file")
                return False

            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
            bits = self._text_to_bits(text)
            print(f"Text to embed: '{text}'")
            print(f"Text converted to {len(bits)} bits")
            print(f"Using embedding strength: {self.strength}")

            blocks_per_frame = ((height // self.block_size) * (width // self.block_size))
            total_capacity = blocks_per_frame * total_frames

            if len(bits) > total_capacity:
                print(f"Error: Text too long! Required: {len(bits)} bits, Available: {total_capacity} bits")
                return False

            print(f"Capacity check: {len(bits)}/{total_capacity} bits ({len(bits)/total_capacity*100:.1f}%)")

            # Choose codec based on lossless preference
            if use_lossless:
                print("Using lossless codec for better steganography...")
                # Try different lossless options
                codecs_to_try = [
                    ('FFV1', cv2.VideoWriter_fourcc(*'FFV1')),  # Lossless
                    ('HFYU', cv2.VideoWriter_fourcc(*'HFYU')),  # Huffman lossless
                    ('MJPG', cv2.VideoWriter_fourcc(*'MJPG')),  # Fallback
                ]
            else:
                codecs_to_try = [
                    ('MJPG', cv2.VideoWriter_fourcc(*'MJPG')),
                    ('XVID', cv2.VideoWriter_fourcc(*'XVID')),
                ]

            out = None
            used_codec = None
            
            for codec_name, fourcc in codecs_to_try:
                print(f"Trying codec: {codec_name}")
                out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height), isColor=False)
                if out.isOpened():
                    used_codec = codec_name
                    print(f"✓ Successfully initialized with {codec_name} codec")
                    break
                else:
                    print(f"✗ {codec_name} codec failed")
                    if out:
                        out.release()

            if not out or not out.isOpened():
                print("Error: Could not create output video file with any codec")
                return False

            print(f"\nEmbedding text using {used_codec} codec...")
            bit_index = 0
            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if bit_index < len(bits):
                    modified_gray, bit_index = self._process_frame_for_embedding(gray_frame, bits, bit_index)
                else:
                    modified_gray = gray_frame

                out.write(modified_gray)
                frame_count += 1

                if frame_count % 30 == 0 or bit_index >= len(bits):
                    print(f"Progress: {frame_count}/{total_frames} frames ({(frame_count/total_frames)*100:.1f}%), "
                          f"bits: {bit_index}/{len(bits)} ({(bit_index/len(bits))*100:.1f}%)")

                if bit_index >= len(bits) and frame_count % 30 != 0:
                    print("All bits embedded! Processing remaining frames...")

            cap.release()
            out.release()

            print(f"\n✓ Successfully created video with embedded text: {output_video_path}")
            print(f"✓ Embedded {bit_index} out of {len(bits)} bits using {used_codec} codec")
            print(f"✓ Embedding strength used: {self.strength}")
            
            if used_codec in ['MJPG', 'XVID']:
                print("\n⚠️  Warning: Used lossy codec which may affect steganography quality")
                print("   Consider using a lossless format if extraction fails")
            
            return True

        except Exception as e:
            print(f"Error during embedding: {e}")
            return False

def main():
    if len(sys.argv) < 4 or len(sys.argv) > 6:
        print("Usage: python embed_text.py <input_video> <output_video> <text_message> [strength] [--lossy]")
        print("\nExamples:")
        print("python embed_text.py input.mp4 output.avi 'Hello World'")
        print("python embed_text.py input.mp4 output.avi 'Hello World' 40.0")
        print("python embed_text.py input.mp4 output.avi 'Hello World' 40.0 --lossy")
        print("\nParameters:")
        print("  strength: Embedding strength (default: 50.0, higher = more robust)")
        print("  --lossy: Use lossy codecs (default: try lossless first)")
        sys.exit(1)

    input_video = sys.argv[1]
    output_video = sys.argv[2]
    text_message = sys.argv[3]
    
    # Optional strength parameter
    strength = 50.0
    use_lossless = True
    
    for i in range(4, len(sys.argv)):
        arg = sys.argv[i]
        if arg == '--lossy':
            use_lossless = False
        else:
            try:
                strength = float(arg)
            except ValueError:
                print(f"Error: Invalid strength value '{arg}'. Must be a number.")
                sys.exit(1)

    if not os.path.exists(input_video):
        print(f"Error: Input video file '{input_video}' not found!")
        sys.exit(1)

    embedder = VideoTextEmbedder(block_size=8, strength=strength)

    print("=== Video Text Embedding (Improved) ===")
    print(f"Input video: {input_video}")
    print(f"Output video: {output_video}")
    print(f"Message: '{text_message}' ({len(text_message)} characters)")
    print(f"Embedding strength: {strength}")
    print(f"Codec preference: {'Lossless' if use_lossless else 'Lossy'}")

    success = embedder.embed_text_in_video(input_video, output_video, text_message, use_lossless)

    if success:
        print(f"\n🎉 Success! Text hidden in video: {output_video}")
        print("You can now extract the hidden message using the updated extract_text.py")
        print(f"\nTo extract: python extract_text.py {output_video}")
    else:
        print(f"\n❌ Failed to embed text in video")
        sys.exit(1)

if __name__ == "__main__":
    main()