#!/usr/bin/env python3
"""
Video Text Embedding Script
Hides text messages in video DCT coefficients.

Usage: python embed.py input_video.mp4 output_video.mp4 "Your secret message"
"""

import cv2
import numpy as np
from scipy.fftpack import dct, idct
import hashlib
import struct
import sys
import os

class VideoTextEmbedder:
    def __init__(self, block_size=8, alpha=0.2):
        """
        Initialize the embedding system.
        
        Args:
            block_size: Size of DCT blocks (8x8 is standard)
            alpha: Embedding strength (0.1-0.5, higher = more robust)
        """
        self.block_size = block_size
        self.alpha = alpha
        self.magic_header = b'VSTEG'
        
    def _text_to_binary(self, text):
        """Convert text to binary string with metadata."""
        text_bytes = text.encode('utf-8')
        length = len(text_bytes)
        checksum = hashlib.md5(text_bytes).digest()[:4]
        
        # Pack: magic header + length + checksum + text
        data = self.magic_header + struct.pack('<I', length) + checksum + text_bytes
        
        # Convert to binary string
        binary = ''.join(format(byte, '08b') for byte in data)
        return binary
    
    def _dct2d(self, block):
        """2D DCT transform."""
        return dct(dct(block.T, norm='ortho').T, norm='ortho')
    
    def _idct2d(self, block):
        """2D inverse DCT transform."""
        return idct(idct(block.T, norm='ortho').T, norm='ortho')
    
    def _embed_bit_in_coefficient(self, coeff, bit):
        """Embed a bit in a DCT coefficient using quantization."""
        quantization_step = self.alpha * abs(coeff) if abs(coeff) > 1 else self.alpha
        
        if bit == '1':
            # Make coefficient represent odd value
            if int(coeff / quantization_step) % 2 == 0:
                coeff += quantization_step if coeff >= 0 else -quantization_step
        else:
            # Make coefficient represent even value
            if int(coeff / quantization_step) % 2 == 1:
                coeff += quantization_step if coeff >= 0 else -quantization_step
        
        return coeff
    
    def _get_embedding_positions(self):
        """Get mid-frequency positions for embedding in 8x8 DCT block."""
        positions = []
        # Use mid-frequency coefficients for robustness
        for i in range(1, 4):  # Skip DC coefficient (0,0)
            for j in range(1, 4):
                if i + j < 5:  # Avoid very high frequencies
                    positions.append((i, j))
        return positions
    
    def embed_text(self, input_video_path, output_video_path, text):
        """Embed text in video using DCT coefficients."""
        print(f"🔐 Embedding text: '{text}'")
        print(f"📥 Input video: {input_video_path}")
        print(f"📤 Output video: {output_video_path}")
        
        # Validate input
        if not os.path.exists(input_video_path):
            raise FileNotFoundError(f"Input video not found: {input_video_path}")
        
        if not text.strip():
            raise ValueError("Text message cannot be empty")
        
        # Convert text to binary
        binary_data = self._text_to_binary(text)
        print(f"📊 Binary data length: {len(binary_data)} bits")
        
        # Open video
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise ValueError("Could not open input video")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"🎬 Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Calculate embedding capacity
        blocks_per_frame = (height // self.block_size) * (width // self.block_size)
        embedding_positions = self._get_embedding_positions()
        bits_per_frame = blocks_per_frame * len(embedding_positions)
        total_capacity = bits_per_frame * total_frames
        
        print(f"💾 Capacity: {bits_per_frame} bits/frame, {total_capacity} total bits")
        
        if len(binary_data) > total_capacity:
            cap.release()
            raise ValueError(f"❌ Text too long! Need {len(binary_data)} bits, capacity is {total_capacity}")
        
        print(f"✅ Capacity check passed ({len(binary_data)}/{total_capacity} bits)")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            cap.release()
            raise ValueError("Could not create output video")
        
        bit_index = 0
        frame_count = 0
        
        print("🔄 Processing frames...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % 30 == 0 or frame_count == 1:
                progress = (frame_count / total_frames) * 100
                print(f"   Frame {frame_count}/{total_frames} ({progress:.1f}%)")
            
            # Convert to YUV (work on luminance channel)
            yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            y_channel = yuv_frame[:, :, 0].astype(np.float32)
            
            # Process 8x8 blocks
            for i in range(0, height - self.block_size + 1, self.block_size):
                for j in range(0, width - self.block_size + 1, self.block_size):
                    if bit_index >= len(binary_data):
                        break
                    
                    # Extract 8x8 block
                    block = y_channel[i:i+self.block_size, j:j+self.block_size]
                    
                    # Apply DCT
                    dct_block = self._dct2d(block)
                    
                    # Embed bits in mid-frequency coefficients
                    for pos in embedding_positions:
                        if bit_index >= len(binary_data):
                            break
                        
                        row, col = pos
                        bit = binary_data[bit_index]
                        dct_block[row, col] = self._embed_bit_in_coefficient(dct_block[row, col], bit)
                        bit_index += 1
                    
                    # Apply inverse DCT
                    modified_block = self._idct2d(dct_block)
                    y_channel[i:i+self.block_size, j:j+self.block_size] = modified_block
                
                if bit_index >= len(binary_data):
                    break
            
            # Convert back to BGR
            yuv_frame[:, :, 0] = np.clip(y_channel, 0, 255).astype(np.uint8)
            modified_frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR)
            
            out.write(modified_frame)
        
        cap.release()
        out.release()
        
        print(f"✅ Successfully embedded {bit_index} bits in {frame_count} frames")
        print(f"💾 Output saved to: {output_video_path}")
        
        # Verify output file was created
        if os.path.exists(output_video_path):
            file_size = os.path.getsize(output_video_path) / (1024 * 1024)  # MB
            print(f"📁 Output file size: {file_size:.1f} MB")
        else:
            print("⚠️  Warning: Output file was not created successfully")

def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) != 4:
        print("Usage: python embed.py <input_video> <output_video> <text_message>")
        print("")
        print("Examples:")
        print('  python embed.py input.mp4 output.mp4 "This is my secret message"')
        print('  python embed.py vacation.mp4 vacation_hidden.mp4 "Meet at cafe 3pm"')
        print("")
        print("Note: Text will be embedded invisibly in the video's DCT coefficients")
        sys.exit(1)
    
    input_video = sys.argv[1]
    output_video = sys.argv[2]
    text_message = sys.argv[3]
    
    try:
        embedder = VideoTextEmbedder(alpha=0.2)
        embedder.embed_text(input_video, output_video, text_message)
        
        print("\n🎉 Embedding completed successfully!")
        print(f"Your message is now hidden in: {output_video}")
        print("Use extract.py to retrieve the message later.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()