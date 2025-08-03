What it does
test1.py is the main embedding script. It takes a video file and secretly embeds text into it by:

Converting video frames to YUV color space
Breaking each frame into 8x8 pixel blocks
Applying DCT transform to each block
Modifying mid-frequency coefficients to encode binary data
Uses a quantization approach where odd/even coefficient values represent 1/0 bits

The text gets encoded with metadata (magic header "VSTEG", length, MD5 checksum) before embedding, so it can be properly extracted later.
test2.py is a debug/testing version that works on a single frame in memory. It embeds text and immediately tries to extract it to verify the process works. Pretty useful for troubleshooting without creating full video files.
Key features

Uses mid-frequency DCT coefficients (avoids DC component and high frequencies)
Embedding strength controlled by alpha parameter (0.2 default)
Includes error checking and capacity validation
Works on luminance channel only for better imperceptibility

The approach is pretty standard for DCT-based steganography. The quantization method for bit embedding is straightforward - just nudge coefficients to make them represent odd/even values based on the bit you want to store.
Capacity depends on video resolution and length. For a typical 1080p video, you can hide quite a bit of text without noticeable quality loss.


Code Implementation Details
Libraries Used
OpenCV (cv2) - Main video processing library

VideoCapture() for reading input videos
VideoWriter() for creating output videos
Color space conversion (COLOR_BGR2YUV, COLOR_YUV2BGR)
Video property extraction (FPS, dimensions, frame count)

NumPy - Array operations and numerical processing

Frame data manipulation as arrays
Block extraction and reconstruction
Data type conversions (astype(), clip())

SciPy - DCT transform functions

scipy.fftpack.dct() and idct() for 2D DCT operations
Using norm='ortho' for orthogonal normalization

Core Python modules:

hashlib - MD5 checksums for data integrity
struct - Binary data packing/unpacking (pack('<I'))
sys - Command line argument handling
os - File system operations and validation

Technical Approach
DCT Block Processing:

Standard 8x8 block size (JPEG-style)
2D DCT applied via transpose method: dct(dct(block.T).T)
Mid-frequency coefficient selection (positions like (1,1), (1,2), (2,1))

Bit Embedding Method:

Quantization-based approach using coefficient modulo operations
alpha parameter controls embedding strength (0.2 = 20% of coefficient magnitude)
Odd quantized values = bit '1', even values = bit '0'

Data Format:

Magic header: b'VSTEG' (5 bytes)
Message length: 32-bit unsigned integer
MD5 checksum: First 4 bytes of hash
UTF-8 encoded text data

Video Processing Pipeline:

YUV color space conversion (embed in luminance only)
Block-wise DCT processing
Sequential bit embedding across frames
Inverse DCT and color space conversion back to BGR
