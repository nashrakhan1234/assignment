#!/usr/bin/env python3
"""
Debug script to test embedding and extraction on the same video in memory
"""

import cv2
import numpy as np
from scipy.fftpack import dct, idct
import hashlib
import struct
import sys

class DebugVideoSteganography:
    def __init__(self, block_size=8, alpha=0.2):
        self.block_size = block_size
        self.alpha = alpha
        self.magic_header = b'VSTEG'
        
    def _text_to_binary(self, text):
        """Convert text to binary string with metadata."""
        text_bytes = text.encode('utf-8')
        length = len(text_bytes)
        checksum = hashlib.md5(text_bytes).digest()[:4]
        
        data = self.magic_header + struct.pack('<I', length) + checksum + text_bytes
        binary = ''.join(format(byte, '08b') for byte in data)
        print(f"Debug: Binary representation of '{text}':")
        print(f"  Magic header: {self.magic_header}")
        print(f"  Length: {length}")
        print(f"  Total binary length: {len(binary)} bits")
        print(f"  First 5 bytes (hex): {data[:5].hex()}")
        return binary
    
    def _binary_to_text(self, binary):
        """Convert binary string back to text with validation."""
        try:
            # Convert binary to bytes
            byte_data = bytearray()
            for i in range(0, len(binary), 8):
                if i + 8 <= len(binary):
                    byte_val = int(binary[i:i+8], 2)
                    byte_data.append(byte_val)
            
            byte_data = bytes(byte_data)
            print(f"Debug: Attempting to decode {len(byte_data)} bytes")
            print(f"  First 5 bytes (hex): {byte_data[:5].hex()}")
            print(f"  Expected magic: {self.magic_header.hex()}")
            
            # Check magic header
            if len(byte_data) < len(self.magic_header):
                print(f"  ❌ Not enough bytes for magic header")
                return None
            if not byte_data.startswith(self.magic_header):
                print(f"  ❌ Magic header mismatch")
                return None
            
            # Extract length
            if len(byte_data) < len(self.magic_header) + 4:
                print(f"  ❌ Not enough bytes for length field")
                return None
            
            length = struct.unpack('<I', byte_data[len(self.magic_header):len(self.magic_header)+4])[0]
            print(f"  ✅ Magic header found, text length: {length}")
            
            # Validate length is reasonable
            if length > 10000 or length == 0:
                print(f"  ❌ Invalid length: {length}")
                return None
            
            # Extract checksum and text
            checksum_start = len(self.magic_header) + 4
            text_start = checksum_start + 4
            text_end = text_start + length
            
            if len(byte_data) < text_end:
                print(f"  ❌ Not enough bytes for full message (need {text_end}, have {len(byte_data)})")
                return None
            
            stored_checksum = byte_data[checksum_start:text_start]
            text_bytes = byte_data[text_start:text_end]
            
            # Verify checksum
            calculated_checksum = hashlib.md5(text_bytes).digest()[:4]
            if stored_checksum != calculated_checksum:
                print(f"  ❌ Checksum mismatch")
                return None
            
            result = text_bytes.decode('utf-8')
            print(f"  ✅ Successfully decoded: '{result}'")
            return result
        except Exception as e:
            print(f"  ❌ Exception during decode: {e}")
            return None
    
    def _dct2d(self, block):
        """2D DCT transform."""
        return dct(dct(block.T, norm='ortho').T, norm='ortho')
    
    def _idct2d(self, block):
        """2D inverse DCT transform."""
        return idct(idct(block.T, norm='ortho').T, norm='ortho')
    
    def _embed_bit_in_coefficient(self, coeff, bit):
        """Embed a bit in a DCT coefficient using quantization."""
        quantization_step = max(self.alpha, 0.1)  # Minimum step to avoid precision issues
        
        if bit == '1':
            # Make coefficient clearly represent odd quantization level
            if abs(coeff) < quantization_step:
                coeff = quantization_step if coeff >= 0 else -quantization_step
            else:
                current_level = int(round(coeff / quantization_step))
                if current_level % 2 == 0:  # Currently even, make it odd
                    coeff += quantization_step if coeff >= 0 else -quantization_step
        else:
            # Make coefficient clearly represent even quantization level (including zero)
            current_level = int(round(coeff / quantization_step))
            if current_level % 2 == 1:  # Currently odd, make it even
                coeff += quantization_step if coeff >= 0 else -quantization_step
        
        return coeff
    
    def _extract_bit_from_coefficient(self, coeff):
        """Extract a bit from a DCT coefficient."""
        quantization_step = max(self.alpha, 0.1)  # Same minimum step as embedding
        
        # For very small coefficients, consider them as 0 (even)
        if abs(coeff) < quantization_step * 0.5:
            return '0'
        
        # Determine if the coefficient represents odd or even quantization level
        quantized_value = int(round(coeff / quantization_step))
        return '1' if quantized_value % 2 == 1 else '0'
    
    def _get_embedding_positions(self):
        """Get mid-frequency positions for embedding in 8x8 DCT block."""
        positions = []
        for i in range(1, 4):  # Skip DC coefficient (0,0)
            for j in range(1, 4):
                if i + j < 5:  # Avoid very high frequencies
                    positions.append((i, j))
        print(f"Debug: Using {len(positions)} embedding positions: {positions}")
        return positions
    
    def test_embed_extract(self, frame, text):
        """Test embedding and extraction on a single frame."""
        print(f"\n=== Testing embed/extract on single frame ===")
        print(f"Frame shape: {frame.shape}")
        
        # Convert text to binary
        binary_data = self._text_to_binary(text)
        
        # Convert frame to YUV and get Y channel
        yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        y_channel = yuv_frame[:, :, 0].astype(np.float32)
        original_y = y_channel.copy()
        
        height, width = y_channel.shape
        embedding_positions = self._get_embedding_positions()
        
        # Calculate how many bits we can embed in first few blocks
        blocks_needed = (len(binary_data) + len(embedding_positions) - 1) // len(embedding_positions)
        print(f"Need {blocks_needed} blocks to embed {len(binary_data)} bits")
        
        # Embed in first few blocks only
        bit_index = 0
        blocks_processed = 0
        
        for i in range(0, height - self.block_size + 1, self.block_size):
            for j in range(0, width - self.block_size + 1, self.block_size):
                if bit_index >= len(binary_data):
                    break
                
                # Extract 8x8 block
                block = y_channel[i:i+self.block_size, j:j+self.block_size]
                
                # Apply DCT
                dct_block = self._dct2d(block)
                
                # Embed bits
                for pos in embedding_positions:
                    if bit_index >= len(binary_data):
                        break
                    
                    row, col = pos
                    bit = binary_data[bit_index]
                    old_coeff = dct_block[row, col]
                    dct_block[row, col] = self._embed_bit_in_coefficient(dct_block[row, col], bit)
                    new_coeff = dct_block[row, col]
                    
                    if blocks_processed < 2:  # Debug first 2 blocks
                        print(f"  Block {blocks_processed}, pos {pos}: bit '{bit}', coeff {old_coeff:.3f} -> {new_coeff:.3f}")
                    
                    bit_index += 1
                
                # Apply inverse DCT
                modified_block = self._idct2d(dct_block)
                y_channel[i:i+self.block_size, j:j+self.block_size] = modified_block
                blocks_processed += 1
            
            if bit_index >= len(binary_data):
                break
        
        print(f"Embedded {bit_index} bits in {blocks_processed} blocks")
        
        # Now extract from the same modified frame
        print(f"\n--- Extracting from modified frame ---")
        extracted_bits = ""
        blocks_checked = 0
        
        for i in range(0, height - self.block_size + 1, self.block_size):
            for j in range(0, width - self.block_size + 1, self.block_size):
                if len(extracted_bits) >= len(binary_data):
                    break
                
                # Extract 8x8 block
                block = y_channel[i:i+self.block_size, j:j+self.block_size]
                
                # Apply DCT
                dct_block = self._dct2d(block)
                
                # Extract bits
                for pos in embedding_positions:
                    if len(extracted_bits) >= len(binary_data):
                        break
                    
                    row, col = pos
                    bit = self._extract_bit_from_coefficient(dct_block[row, col])
                    extracted_bits += bit
                    
                    if blocks_checked < 2:  # Debug first 2 blocks
                        coeff = dct_block[row, col]
                        print(f"  Block {blocks_checked}, pos {pos}: coeff {coeff:.3f} -> bit '{bit}'")
                
                blocks_checked += 1
            
            if len(extracted_bits) >= len(binary_data):
                break
        
        print(f"Extracted {len(extracted_bits)} bits from {blocks_checked} blocks")
        
        # Compare original vs extracted
        print(f"\n--- Comparison ---")
        print(f"Original binary length:  {len(binary_data)}")
        print(f"Extracted binary length: {len(extracted_bits)}")
        
        # Truncate extracted to match original length
        if len(extracted_bits) > len(binary_data):
            extracted_bits = extracted_bits[:len(binary_data)]
        
        print(f"Original binary:  {binary_data[:50]}...")
        print(f"Extracted binary: {extracted_bits[:50]}...")
        exact_match = binary_data == extracted_bits
        print(f"Match: {exact_match}")
        
        if not exact_match:
            # Find first difference
            for i, (orig, extr) in enumerate(zip(binary_data, extracted_bits)):
                if orig != extr:
                    print(f"First difference at bit {i}: original '{orig}' vs extracted '{extr}'")
                    # Calculate which block and position this corresponds to
                    block_num = i // len(embedding_positions)
                    pos_in_block = i % len(embedding_positions)
                    pos = embedding_positions[pos_in_block]
                    print(f"  This is block {block_num}, position {pos}")
                    break
        
        # Try to decode
        extracted_text = self._binary_to_text(extracted_bits)
        
        return extracted_text

def main():
    if len(sys.argv) != 2:
        print("Usage: python debug.py video_file.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    test_text = "hello world"
    
    # Load first frame
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Could not open video")
        sys.exit(1)
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Could not read frame")
        sys.exit(1)
    
    # Test with different alpha values
    alphas_to_test = [0.2, 0.5, 1.0]
    
    for alpha in alphas_to_test:
        print(f"\n{'='*20} Testing with alpha={alpha} {'='*20}")
        debug_system = DebugVideoSteganography(alpha=alpha)
        result = debug_system.test_embed_extract(frame, test_text)
        
        print(f"Alpha {alpha}: {'SUCCESS' if result == test_text else 'FAILED'}")
        
        if result == test_text:
            print(f"✅ Found working alpha: {alpha}")
            break

if __name__ == "__main__":
    main()
