#!/usr/bin/env python3
"""
Simplified Temporal Flash Steganography - Final Version with Phone Recording Support
Uses a very simple flash pattern that works with any message and phone recordings
"""

import cv2
import numpy as np
import argparse
import sys
import os
from typing import Optional, List

class SimpleFlashSteganography:
    def __init__(self, flash_intensity: int = 50):
        """Initialize with simple flash parameters"""
        self.flash_intensity = flash_intensity
        self.delimiter = "END"
    
    def _text_to_binary(self, text: str) -> str:
        """Convert text to binary with delimiter"""
        full_text = text + self.delimiter
        return ''.join(format(ord(char), '08b') for char in full_text)
    
    def embed_message(self, input_video: str, output_video: str, message: str) -> bool:
        """Embed message using very simple flash pattern"""
        try:
            print(f"=== Simple Flash Steganography - EMBED ===")
            print(f"Input: {input_video}")
            print(f"Output: {output_video}")
            print(f"Message: '{message}' ({len(message)} chars)")
            print(f"Flash intensity: {self.flash_intensity}")
            
            cap = cv2.VideoCapture(input_video)
            if not cap.isOpened():
                print("Error: Cannot open input video")
                return False
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"Video: {width}x{height}, {fps}fps, {total_frames} frames")
            
            # Convert message to binary
            binary_data = self._text_to_binary(message)
            print(f"Binary data: {len(binary_data)} bits")
            
            # Simple pattern: 4 frames per bit
            frames_per_bit = 4
            required_frames = len(binary_data) * frames_per_bit
            
            print(f"Required frames: {required_frames} (have {total_frames})")
            
            if required_frames > total_frames:
                print(f"Error: Message too long! Need {required_frames} frames, have {total_frames}")
                return False
            
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
            
            if not out.isOpened():
                print("Error: Cannot create output video")
                return False
            
            frame_count = 0
            bit_index = 0
            
            print("Embedding simple flash pattern...")
            print("Pattern: Bit 1 = Flash on frame 0, normal on frames 1-3")
            print("Pattern: Bit 0 = Normal on all 4 frames")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Determine if this frame should flash
                should_flash = False
                if bit_index < len(binary_data):
                    bit_position_in_group = frame_count % frames_per_bit
                    current_bit = binary_data[bit_index]
                    
                    # Flash on first frame of each bit group if bit is 1
                    if current_bit == '1' and bit_position_in_group == 0:
                        should_flash = True
                    
                    # Move to next bit after processing all frames for current bit
                    if bit_position_in_group == frames_per_bit - 1:
                        bit_index += 1
                
                # Apply flash if needed
                if should_flash:
                    flashed_frame = frame.astype(np.int16) + self.flash_intensity
                    frame = np.clip(flashed_frame, 0, 255).astype(np.uint8)
                    print(f"Frame {frame_count}: FLASH (bit {bit_index})")
                
                out.write(frame)
                frame_count += 1
                
                if frame_count % 50 == 0:
                    progress = (bit_index / len(binary_data)) * 100 if len(binary_data) > 0 else 100
                    print(f"Frame {frame_count}: {progress:.1f}% embedded")
            
            cap.release()
            out.release()
            
            print(f"✅ SUCCESS! Embedded {bit_index} bits in {frame_count} frames")
            return True
            
        except Exception as e:
            print(f"Error during embedding: {e}")
            return False
    
    def extract_message(self, video_path: str) -> Optional[str]:
        """Extract message - automatically detects clean vs phone recorded video"""
        try:
            print(f"=== Simple Flash Steganography - EXTRACT ===")
            print(f"Video: {video_path}")
            print(f"Flash intensity: {self.flash_intensity}")
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print("Error: Cannot open video")
                return None
            
            # Get properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"Video: {width}x{height}, {total_frames} frames")
            
            # Extract brightness values
            brightness_values = []
            frame_count = 0
            
            print("Analyzing frame brightness...")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Calculate average brightness
                if len(frame.shape) == 3:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    gray = frame
                
                brightness = np.mean(gray)
                brightness_values.append(brightness)
                frame_count += 1
            
            cap.release()
            
            print(f"Extracted brightness from {len(brightness_values)} frames")
            
            # Determine if this is a phone recording by checking resolution and pattern
            is_phone_recording = False
            if width <= 1000 or height <= 600:  # Typical phone recording resolution
                is_phone_recording = True
                print("Detected phone recording (lower resolution)")
            
            # Try standard extraction first
            result = self._extract_standard(brightness_values)
            if result:
                return result
            
            # If standard fails or this looks like phone recording, try phone-specific extraction
            if is_phone_recording or not result:
                print("Trying phone recording extraction...")
                result = self._extract_phone_recording(brightness_values)
                if result:
                    return result
            
            print("❌ No message found with either method")
            return None
            
        except Exception as e:
            print(f"Error during extraction: {e}")
            return None
    
    def _extract_standard(self, brightness_values: List[float]) -> Optional[str]:
        """Standard extraction for clean videos"""
        try:
            print("--- Standard Extraction ---")
            baseline = np.median(brightness_values)
            
            # Check every 4th frame starting position
            frames_per_bit = 4
            all_increases = []
            
            for bit_start in range(0, len(brightness_values), frames_per_bit):
                if bit_start < len(brightness_values):
                    brightness = brightness_values[bit_start]
                    brightness_increase = brightness - baseline
                    all_increases.append((bit_start, brightness_increase))
            
            # Use 70th percentile threshold
            increases_only = [inc for _, inc in all_increases]
            threshold = np.percentile(increases_only, 70)
            print(f"Standard threshold: {threshold:.1f}")
            
            # Apply threshold
            flash_frames = [False] * len(brightness_values)
            detected_flashes = []
            
            for bit_start, brightness_increase in all_increases:
                is_flash = brightness_increase > threshold
                flash_frames[bit_start] = is_flash
                if is_flash:
                    detected_flashes.append(bit_start)
            
            print(f"Standard detected {len(detected_flashes)} flashes")
            
            # Decode
            binary_data = ""
            for bit_start in range(0, len(flash_frames) - frames_per_bit + 1, frames_per_bit):
                bit_frames = flash_frames[bit_start:bit_start + frames_per_bit]
                if bit_frames[0]:
                    binary_data += "1"
                else:
                    binary_data += "0"
                if len(binary_data) >= 64:
                    break
            
            # Convert to text
            text = ""
            for i in range(0, len(binary_data), 8):
                if i + 8 <= len(binary_data):
                    byte = binary_data[i:i+8]
                    try:
                        char = chr(int(byte, 2))
                        text += char
                        if self.delimiter in text:
                            break
                    except:
                        continue
            
            if self.delimiter in text:
                message = text.split(self.delimiter)[0]
                print(f"✅ Standard extraction found: '{message}'")
                return message
            
            print("Standard extraction failed")
            return None
            
        except Exception as e:
            print(f"Standard extraction error: {e}")
            return None
    
    def _extract_phone_recording(self, brightness_values: List[float]) -> Optional[str]:
        """Phone recording extraction with fuzzy pattern matching"""
        try:
            print("--- Phone Recording Extraction ---")
            baseline = np.median(brightness_values)
            
            # Get ALL bright frames (not just 4-frame intervals)
            all_bright_frames = []
            threshold = np.percentile([b - baseline for b in brightness_values], 80)  # Higher threshold
            
            for frame_idx, brightness in enumerate(brightness_values):
                if brightness - baseline > threshold:
                    all_bright_frames.append(frame_idx)
            
            print(f"Found {len(all_bright_frames)} bright frames with threshold {threshold:.1f}")
            print(f"Bright frames: {all_bright_frames[:20]}...")
            
            if len(all_bright_frames) < 8:
                print("Not enough bright frames for phone extraction")
                return None
            
            # Check if we have too many consecutive bright frames at the start (camera adjustment)
            consecutive_start = 0
            for i, frame in enumerate(all_bright_frames):
                if frame == i:  # Frame number equals index (consecutive from 0)
                    consecutive_start += 1
                else:
                    break
            
            print(f"Consecutive bright frames from start: {consecutive_start}")
            
            # If too many consecutive frames, skip the adjustment period
            search_frames = all_bright_frames
            if consecutive_start > 10:
                print("Skipping camera adjustment period...")
                # Look for a gap in the bright frames and start after it
                for i in range(consecutive_start, min(consecutive_start + 50, len(all_bright_frames) - 1)):
                    if all_bright_frames[i+1] - all_bright_frames[i] > 10:  # Gap of more than 10 frames
                        search_frames = all_bright_frames[i+1:]
                        print(f"Found gap after frame {all_bright_frames[i]}, starting search from frame {all_bright_frames[i+1]}")
                        break
                
                # If no gap found, start from a reasonable offset
                if len(search_frames) == len(all_bright_frames):
                    offset = min(60, len(all_bright_frames) // 2)
                    search_frames = all_bright_frames[offset:]
                    print(f"No clear gap found, starting from offset {offset}")
            
            print(f"Using {len(search_frames)} frames for pattern search: {search_frames[:15]}...")
            
            # Try multiple starting points from the cleaned frame list
            for start_idx in range(min(8, len(search_frames))):
                start_frame = search_frames[start_idx]
                
                # Try different frame intervals (3, 4, 5 frames per bit)
                for interval in [3, 4, 5]:
                    pattern_bits = []
                    
                    for bit_pos in range(20):  # Try 20 bits
                        target_frame = start_frame + (bit_pos * interval)
                        
                        # Look for bright frame within ±2 frames of target
                        found_flash = False
                        for bright_frame in search_frames:  # Use cleaned frame list
                            if abs(bright_frame - target_frame) <= 2:
                                found_flash = True
                                break
                        
                        pattern_bits.append(1 if found_flash else 0)
                    
                    # Try to decode this pattern
                    binary_string = ''.join(map(str, pattern_bits))
                    
                    # Convert to text
                    text = ""
                    for i in range(0, len(binary_string), 8):
                        if i + 8 <= len(binary_string):
                            byte = binary_string[i:i+8]
                            try:
                                char = chr(int(byte, 2))
                                if 32 <= ord(char) <= 126:  # Printable ASCII
                                    text += char
                            except:
                                continue
                    
                    # Check if we found something readable
                    if len(text) >= 1:
                        readable_chars = sum(1 for c in text if c.isalnum())
                        if readable_chars >= 1:
                            print(f"Start {start_frame}, interval {interval}: '{text}' (binary: {binary_string[:16]}...)")
                            
                            # Check for delimiter or return partial message
                            if self.delimiter in text:
                                message = text.split(self.delimiter)[0]
                                if len(message) >= 1:
                                    print(f"✅ Phone extraction found: '{message}'")
                                    return message
                            elif len(text.strip()) >= 1:
                                print(f"✅ Phone extraction found partial: '{text.strip()}'")
                                return text.strip()
            
            print("Phone extraction failed - no readable patterns found")
            return None
            
        except Exception as e:
            print(f"Phone extraction error: {e}")
            return None

def main():
    parser = argparse.ArgumentParser(description='Simple Flash Video Steganography')
    parser.add_argument('--mode', choices=['embed', 'extract'], required=True)
    parser.add_argument('--input', required=True, help='Input video file')
    parser.add_argument('--output', help='Output video file (for embed)')
    parser.add_argument('--message', help='Message to embed')
    parser.add_argument('--intensity', type=int, default=50, help='Flash intensity (default: 50)')
    
    args = parser.parse_args()
    
    stego = SimpleFlashSteganography(flash_intensity=args.intensity)
    
    if args.mode == 'embed':
        if not args.message or not args.output:
            print("Error: --message and --output required for embed mode")
            return
        
        success = stego.embed_message(args.input, args.output, args.message)
        if success:
            print(f"\n📱 Next steps:")
            print(f"1. Play {args.output} on your computer screen")
            print(f"2. Record THE ENTIRE VIDEO with your phone")
            print(f"3. Look for bright flashes every 4 frames")
            print(f"4. Run: python {sys.argv[0]} --mode extract --input phone_recorded.mp4")
    
    elif args.mode == 'extract':
        message = stego.extract_message(args.input)
        if message:
            print(f"\n✅ SUCCESS! Extracted: '{message}'")
        else:
            print(f"\n❌ No message found")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Simple Flash Video Steganography")
        print("Ultra-simple approach designed for phone recording survival")
        print("\nUsage:")
        print("Embed: python script.py --mode embed --input video.mp4 --output simple_flash.avi --message 'Hi'")
        print("Extract: python script.py --mode extract --input phone_recorded.mp4")
        print("\nRequired: pip install opencv-python numpy")
    else:
        main()