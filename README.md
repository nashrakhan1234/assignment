(base) C:\Users\nashr\OneDrive\Desktop\assignments>python extract_text_1.py output_lossless.avi 75.0
=== Video Text Extraction (CORRECTED) ===
Video file: output_lossless.avi
Extraction strength: 75.0
Opening video: output_lossless.avi
Video properties: 1920x1080, 18 FPS, 199 frames
Using extraction strength: 75.0
Processing all 199 frames

Extracting bits from video frames...
✓ Found valid text in first frame!

🎉 SUCCESS! Hidden message found:
📝 "Hello tick tick"

Message length: 15 characters

(base) C:\Users\nashr\OneDrive\Desktop\assignments>python extract_text_1.py input.mp4 75.0
=== Video Text Extraction (CORRECTED) ===
Video file: input.mp4
Extraction strength: 75.0
Opening video: input.mp4
Video properties: 1920x1080, 18 FPS, 199 frames
Using extraction strength: 75.0
Processing all 199 frames

Extracting bits from video frames...
Progress: 30/199 frames (15.1%)
Progress: 60/199 frames (30.2%)
Progress: 90/199 frames (45.2%)
Progress: 120/199 frames (60.3%)
Progress: 150/199 frames (75.4%)
Progress: 180/199 frames (90.5%)

Extracted 6447600 total bits from 199 frames
First 64 bits: 0000000000000000000000000000000000000000000000000000000000000000
First 8 bytes (hex): 00 00 00 00 00 00 00 00
Expected magic (hex): 44435453544547
Decoding bits to text...
❌ No valid hidden text found

--- Normal extraction failed, trying different thresholds ---
Trying different threshold multipliers: [0.6, 0.5, 0.7, 0.4, 0.8, 0.3, 0.9, 0.2]

--- Trying threshold = strength * 0.6 ---
Opening video: input.mp4
Video properties: 1920x1080, 18 FPS, 199 frames
Using extraction strength: 75.0
Processing first 2 frames

Extracting bits from video frames...

Extracted 64800 total bits from 2 frames
First 64 bits: 0000000000000000000000000000000000000000000000000000000000000000
First 8 bytes (hex): 00 00 00 00 00 00 00 00
Expected magic (hex): 44435453544547
Decoding bits to text...
❌ No valid hidden text found

--- Trying threshold = strength * 0.5 ---
Opening video: input.mp4
Video properties: 1920x1080, 18 FPS, 199 frames
Using extraction strength: 75.0
Processing first 2 frames

Extracting bits from video frames...

Extracted 64800 total bits from 2 frames
First 64 bits: 0000000000000000000000000000000000000000000000000000000000000000
First 8 bytes (hex): 00 00 00 00 00 00 00 00
Expected magic (hex): 44435453544547
Decoding bits to text...
❌ No valid hidden text found

--- Trying threshold = strength * 0.7 ---
Opening video: input.mp4
Video properties: 1920x1080, 18 FPS, 199 frames
Using extraction strength: 75.0
Processing first 2 frames

Extracting bits from video frames...

Extracted 64800 total bits from 2 frames
First 64 bits: 0000000000000000000000000000000000000000000000000000000000000000
First 8 bytes (hex): 00 00 00 00 00 00 00 00
Expected magic (hex): 44435453544547
Decoding bits to text...
❌ No valid hidden text found

--- Trying threshold = strength * 0.4 ---
Opening video: input.mp4
Video properties: 1920x1080, 18 FPS, 199 frames
Using extraction strength: 75.0
Processing first 2 frames

Extracting bits from video frames...

Extracted 64800 total bits from 2 frames
First 64 bits: 0000000000000000000000000000000000000000000000000000000000100000
First 8 bytes (hex): 00 00 00 00 00 00 00 20
Expected magic (hex): 44435453544547
Decoding bits to text...
❌ No valid hidden text found

--- Trying threshold = strength * 0.8 ---
Opening video: input.mp4
Video properties: 1920x1080, 18 FPS, 199 frames
Using extraction strength: 75.0
Processing first 2 frames

Extracting bits from video frames...

Extracted 64800 total bits from 2 frames
First 64 bits: 0000000000000000000000000000000000000000000000000000000000000000
First 8 bytes (hex): 00 00 00 00 00 00 00 00
Expected magic (hex): 44435453544547
Decoding bits to text...
❌ No valid hidden text found

--- Trying threshold = strength * 0.3 ---
Opening video: input.mp4
Video properties: 1920x1080, 18 FPS, 199 frames
Using extraction strength: 75.0
Processing first 2 frames

Extracting bits from video frames...

Extracted 64800 total bits from 2 frames
First 64 bits: 0000000000000010000010000000000000000001000000000000000010100000
First 8 bytes (hex): 00 02 08 00 01 00 00 a0
Expected magic (hex): 44435453544547
Decoding bits to text...
❌ No valid hidden text found

--- Trying threshold = strength * 0.9 ---
Opening video: input.mp4
Video properties: 1920x1080, 18 FPS, 199 frames
Using extraction strength: 75.0
Processing first 2 frames

Extracting bits from video frames...

Extracted 64800 total bits from 2 frames
First 64 bits: 0000000000000000000000000000000000000000000000000000000000000000
First 8 bytes (hex): 00 00 00 00 00 00 00 00
Expected magic (hex): 44435453544547
Decoding bits to text...
❌ No valid hidden text found

--- Trying threshold = strength * 0.2 ---
Opening video: input.mp4
Video properties: 1920x1080, 18 FPS, 199 frames
Using extraction strength: 75.0
Processing first 2 frames

Extracting bits from video frames...

Extracted 64800 total bits from 2 frames
First 64 bits: 0000000001000010010010000000000000000101100000000000100010100000
First 8 bytes (hex): 00 42 48 00 05 80 08 a0
Expected magic (hex): 44435453544547
Decoding bits to text...
❌ No valid hidden text found
❌ Failed with all threshold values

❌ No hidden text could be extracted from this video

Troubleshooting:
• Check that the embedding strength matches (75.0)
• Ensure the video wasn't re-encoded or compressed
• Try different strength values manually

(base) C:\Users\nashr\OneDrive\Desktop\assignments>
