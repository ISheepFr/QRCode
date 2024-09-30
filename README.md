# QR Code Generator & Steganography with Voronoi Diagrams

## Description

This program allows you to generate QR codes, hide them within host images using Voronoi diagrams, and extract them later. It also includes features for hiding and extracting multiple QR codes from a single host image. The program provides an interactive menu with various options to manipulate the generated and augmented QR codes.

## Features

The program includes the following functionalities:

1. **Generate a QR Code**: Create a QR code with a custom message and save it as a `.png` file. <br><br>
<img src="img/1_normal_qr.png" width="200" height="200" alt="Generated normal QR Code"></img> &nbsp;&nbsp; <img src="img/2_secret_qr.png" width="200" height="200" alt="Generated secret QR Code"></img> <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*Normal QR* &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; *Secret QR*

3. **Hide a QR Code in a Host Image**: Select a generated QR code and hide it within a host image using Voronoi diagrams, generating a key file for extraction.<br><br>
<img src="img/2_hide_qr.png" height="200" alt="option hide"></img> &nbsp;&nbsp; <img src="img/2_dissimulated_qr.png" width="200" height="200" alt="hidden_qr"></img> <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*Command menu* &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; *Augmented QR*

5. **Extract a QR Code from a Host Image**: Extract a hidden QR code from a host image using the key file. <br><br>
<img src="img/3_menu.png" height="200" alt="option extract"></img> <br><br> &nbsp;&nbsp; <img src="img/3_extract.png" height="200" alt="extract process"></img> <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; *Final extracted QR*

7. **Hide Multiple QR Codes in a Host Image**: Hide multiple QR codes inside a single host image and generate the key file for extraction.
8. **Extract Multiple QR Codes from a Host Image**
9. **Voronoi Diagram – Hide Multiple QR Codes**: Hide multiple QR codes in a host image using Voronoi diagrams, providing additional control over the process.
10. **Voronoi Diagram – Extract Multiple QR Codes**: Extract multiple QR codes hidden using Voronoi diagrams from a host image.

## Requirements

To run the program, the following Python libraries are required:

- `opencv-python`
- `qrcode`
- `numpy`

You can install them using `pip`:

```bash
pip install opencv-python qrcode numpy
```
## Directory Structure

The first launch create 3 directories with the following roles :

- `generated_qr/`: Contains the generated QR codes.
- `augmented_qr/`: Contains the host images with hidden QR codes and the corresponding key files for extraction.
- `extracted_qr/`: Contains the extracted QR codes.

## Voronoi Keys

When hiding QR codes using Voronoi diagrams, the program generates a key file corresponding to the embedded QR code. This key file is stored in the augmented_qr directory and contains information necessary for extracting the hidden QR codes later. Each key file includes:

- Coordinates of Germs: The positions used for embedding the QR codes.
- Number of Germs: The total count of germs used in the Voronoi diagram for the specific embedding.
This allows for precise extraction of the hidden QR codes when using the corresponding key file.
