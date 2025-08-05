#!/usr/bin/env python3
"""
Video DCT Steganography - Text Extraction Script (CORRECTED)
This version properly matches the improved embedding algorithm.
"""

import cv2
import numpy as np
from scipy.fftpack import dct
import struct
import sys
import os
from typing import Tuple, Optional

class VideoTextExtractor:
    def __init__(self, block_size: int = 8, strength: float = 75.0):
        """
        Initialize the text extractor.
        
        Args:
            block_size: Size of DCT blocks (must match embedding)
            strength: Embedding strength (must match embedding)
        """
        self.block_size = block_size
        self.strength = strength
        self.magic_header = b'DCTSTEG'  # Magic bytes to identify embedded data
        
    def _apply_dct_2d(self, block: np.ndarray) -> np.ndarray:
        """Apply 2D DCT to a block."""
        return dct(dct(block.T, norm='ortho').T, norm='ortho')
    
    def _extract_bit_from_dct_block(self, dct_block: np.ndarray, pos: Tuple[int, int] = (1, 2)) -> int:
        """
        Extract a bit from a DCT block.
        This matches the improved embedding algorithm exactly.
        
        Embedding logic was:
        - bit == 1: coeff = abs(coeff) + strength (always positive, large)
        - bit == 0: coeff = abs(coeff) - strength*0.8 OR -abs(coeff) - strength*0.5 (small positive or negative)
        
        Args:
            dct_block: 8x8 DCT block
            pos: Position in DCT block to read from
            
        Returns:
            Extracted bit (0 or 1)
        """
        coeff = dct_block[pos[0], pos[1]]
        
        # The embedding creates a clear distinction:
        # bit = 1: Large positive values (original + strength)
        # bit = 0: Small positive or negative values
        
        # Use a threshold that's between the two ranges
        threshold = self.strength * 0.6  # Midpoint between the embedding ranges
        
        if coeff > threshold:
            return 1  # Large positive coefficient -> bit = 1
        else:
            return 0  # Small or negative coefficient -> bit = 0
    
    def _process_frame_for_extraction(self, frame: np.ndarray) -> str:
        """Extract bits from all DCT blocks in a frame."""
        height, width = frame.shape
        extracted_bits = []
        
        # Process blocks in the same order as embedding
        for i in range(0, height - self.block_size + 1, self.block_size):
            for j in range(0, width - self.block_size + 1, self.block_size):
                # Extract block and apply DCT
                block = frame[i:i+self.block_size, j:j+self.block_size].astype(np.float32)
                dct_block = self._apply_dct_2d(block)
                
                # Extract bit
                bit = self._extract_bit_from_dct_block(dct_block)
                extracted_bits.append(str(bit))
        
        return ''.join(extracted_bits)
    
    def _bits_to_text(self, bits: str) -> Optional[str]:
        """
        Convert binary string back to text.
        
        Args:
            bits: Binary string
            
        Returns:
            Decoded text or None if invalid
        """
        try:
            # Convert bits to bytes
            byte_data = bytearray()
            for i in range(0, len(bits), 8):
                if i + 8 <= len(bits):
                    byte_val = int(bits[i:i+8], 2)
                    byte_data.append(byte_val)
            
            # Check for minimum required length
            if len(byte_data) < len(self.magic_header) + 4:
                return None
                
            # Verify magic header
            magic = bytes(byte_data[:len(self.magic_header)])
            if magic != self.magic_header:
                return None
            
            # Extract length
            length_bytes = byte_data[len(self.magic_header):len(self.magic_header)+4]
            length = struct.unpack('<I', bytes(length_bytes))[0]
            
            # Validate length
            start_idx = len(self.magic_header) + 4
            if len(byte_data) < start_idx + length:
                return None
                
            # Extract and decode text
            text_bytes = bytes(byte_data[start_idx:start_idx + length])
            return text_bytes.decode('utf-8')
            
        except Exception as e:
            return None
    
    def extract_text_from_video(self, video_path: str, max_frames: Optional[int] = None) -> Optional[str]:
        """
        Extract hidden text from video file.
        
        Args:
            video_path: Path to video with embedded text
            max_frames: Maximum frames to process (None for all frames)
            
        Returns:
            Extracted text or None if failed
        """
        try:
            print(f"Opening video: {video_path}")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Could not open video file")
                return None
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
            print(f"Using extraction strength: {self.strength}")
            
            # Limit frames if specified
            if max_frames:
                frames_to_process = min(total_frames, max_frames)
                print(f"Processing first {frames_to_process} frames")
            else:
                frames_to_process = total_frames
                print(f"Processing all {frames_to_process} frames")
            
            # Extract bits from frames
            all_bits = []
            frame_count = 0
            
            print("\nExtracting bits from video frames...")
            while frame_count < frames_to_process:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to grayscale
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Extract bits from frame
                frame_bits = self._process_frame_for_extraction(gray_frame)
                all_bits.append(frame_bits)
                
                frame_count += 1
                
                # Progress update
                if frame_count % 30 == 0:
                    progress = (frame_count / frames_to_process) * 100
                    print(f"Progress: {frame_count}/{frames_to_process} frames ({progress:.1f}%)")
                
                # Early termination - try to decode after first frame since message is small
                if frame_count == 1:
                    combined_bits = ''.join(all_bits)
                    test_text = self._bits_to_text(combined_bits)
                    if test_text is not None:
                        print(f"✓ Found valid text in first frame!")
                        cap.release()
                        return test_text
                elif frame_count <= 5:  # Check first few frames individually
                    combined_bits = ''.join(all_bits)
                    test_text = self._bits_to_text(combined_bits)
                    if test_text is not None:
                        print(f"✓ Found valid text at frame {frame_count}!")
                        cap.release()
                        return test_text
            
            cap.release()
            
            # Final attempt with all bits
            combined_bits = ''.join(all_bits)
            print(f"\nExtracted {len(combined_bits)} total bits from {frame_count} frames")
            
            # Debug: Show first few bits and bytes
            if len(combined_bits) >= 64:
                first_bits = combined_bits[:64]
                print(f"First 64 bits: {first_bits}")
                
                first_bytes = []
                for i in range(0, 64, 8):
                    byte_val = int(first_bits[i:i+8], 2)
                    first_bytes.append(f"{byte_val:02x}")
                print(f"First 8 bytes (hex): {' '.join(first_bytes)}")
                print(f"Expected magic (hex): {self.magic_header.hex()}")
            
            # Convert bits back to text
            print("Decoding bits to text...")
            extracted_text = self._bits_to_text(combined_bits)
            
            if extracted_text:
                print("✓ Successfully extracted hidden text!")
                return extracted_text
            else:
                print("❌ No valid hidden text found")
                return None
                
        except Exception as e:
            print(f"Error during extraction: {e}")
            return None
    
    def extract_with_different_thresholds(self, video_path: str) -> Optional[str]:
        """
        Try extraction with different threshold multipliers.
        """
        threshold_multipliers = [0.6, 0.5, 0.7, 0.4, 0.8, 0.3, 0.9, 0.2]
        
        print(f"Trying different threshold multipliers: {threshold_multipliers}")
        
        for multiplier in threshold_multipliers:
            print(f"\n--- Trying threshold = strength * {multiplier} ---")
            
            # Temporarily modify the extraction method
            original_extract = self._extract_bit_from_dct_block
            
            def custom_extract(dct_block, pos=(1, 2)):
                coeff = dct_block[pos[0], pos[1]]
                threshold = self.strength * multiplier
                return 1 if coeff > threshold else 0
            
            self._extract_bit_from_dct_block = custom_extract
            
            result = self.extract_text_from_video(video_path, max_frames=2)  # Test first 2 frames
            if result:
                print(f"✓ Success with threshold multiplier = {multiplier}")
                self._extract_bit_from_dct_block = original_extract  # Restore
                return result
            
            self._extract_bit_from_dct_block = original_extract  # Restore
        
        print("❌ Failed with all threshold values")
        return None


def main():
    """Main function to handle command line arguments."""
    
    if len(sys.argv) < 2 or len(sys.argv) > 4:
        print("Usage: python extract_text.py <video_file> [strength] [max_frames]")
        print("\nExamples:")
        print("python extract_text.py output_lossless.avi")
        print("python extract_text.py output_lossless.avi 75.0")
        print("python extract_text.py output_lossless.avi 75.0 10")
        sys.exit(1)
    
    video_file = sys.argv[1]
    strength = 75.0  # Default strength matching the embedding
    max_frames = None
    
    # Parse optional arguments
    if len(sys.argv) >= 3:
        try:
            strength = float(sys.argv[2])
        except ValueError:
            print("Error: strength must be a number")
            sys.exit(1)
    
    if len(sys.argv) == 4:
        try:
            max_frames = int(sys.argv[3])
        except ValueError:
            print("Error: max_frames must be a number")
            sys.exit(1)
    
    # Check if video file exists
    if not os.path.exists(video_file):
        print(f"Error: Video file '{video_file}' not found!")
        sys.exit(1)
    
    # Initialize extractor with matching strength
    extractor = VideoTextExtractor(block_size=8, strength=strength)
    
    print("=== Video Text Extraction (CORRECTED) ===")
    print(f"Video file: {video_file}")
    print(f"Extraction strength: {strength}")
    if max_frames:
        print(f"Max frames: {max_frames}")
    
    # Try normal extraction first
    extracted_text = extractor.extract_text_from_video(video_file, max_frames)
    
    # If failed, try different thresholds
    if not extracted_text:
        print("\n--- Normal extraction failed, trying different thresholds ---")
        extracted_text = extractor.extract_with_different_thresholds(video_file)
    
    # Display results
    if extracted_text:
        print(f"\n🎉 SUCCESS! Hidden message found:")
        print(f"📝 \"{extracted_text}\"")
        print(f"\nMessage length: {len(extracted_text)} characters")
    else:
        print(f"\n❌ No hidden text could be extracted from this video")
        print("\nTroubleshooting:")
        print("• Check that the embedding strength matches (75.0)")
        print("• Ensure the video wasn't re-encoded or compressed")
        print("• Try different strength values manually")


if __name__ == "__main__":
    main()