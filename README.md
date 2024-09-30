# Voronoi QR Code Steganography Project

## Overview
This project implements a steganographic method for hiding QR codes using Voronoi diagrams. It allows you to:
- Create QR codes with custom messages.
- Hide a secret QR code inside a host QR code using a Voronoi partitioning algorithm.
- Extract hidden QR codes from augmented QR images.

## Features
- **Generate QR Codes**: Create a QR code with any message you choose.
- **Hide QR Codes**: Steganographically hide a secret QR code within a host QR code using Voronoi diagrams.
- **Extract Hidden QR Codes**: Extract the hidden QR code from an augmented QR image.
- **Voronoi Diagrams**: Uses Voronoi diagrams to partition the image and determine where and how the secret QR code is embedded.

## How It Works
1. **Voronoi Diagrams**: The image is divided into regions (called germs) using Voronoi partitioning. Each region is associated with a specific color or pixel range from the host QR code.
2. **Steganographic Key**: A key is generated based on Voronoi regions, which determines how the secret QR code is hidden. The key ensures that pixels from the host QR code and the secret QR code interact in a specific pattern, making the secret QR code retrievable.
3. **Embedding the QR Code**: For each pixel in the host QR code, the algorithm uses the Voronoi region’s key to determine how to hide the corresponding pixel of the secret QR code. The result is a new "augmented" QR code that looks like the host but secretly contains the hidden QR code.
4. **Extracting the Hidden QR Code**: Using the same Voronoi key, the hidden QR code is extracted from the augmented QR image.

## Dependencies
You can install the required dependencies with the following command:
```bash
pip install opencv-python numpy qrcode[pil]
