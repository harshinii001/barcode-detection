# barcode-detection

## Step 1: Text detection
1. text_detection.ipynb detects the text using keras-ocr - (Texts are being the major noises in the barcode detection)
2. The output of text_detection.ipynb is saved in the output1_text_detection folder
3. text_detection_filling.py - converts the images in the output1_text_detection folder to binary and with black colour filling in the bounded area. The output of this file is in output2_text_detection_floodfill. These images can be directly used for the bitwise operation to remove the noise.

## Step 2: Object detection. 
1. By detecting the object boundary, the barcode of all the objects can be detected separately, which helps in detection of repeated barcodes, and by applying the barcode_detection.py on the detected object area barcode can be found more accurately. 
2. background_removal.py - outputs (output3_background_removal) the images with shadows and partially removed object colour  
3. output of background_removal.py is the input of object_edge_detection.py. Output of object_edge_detection.py is stored in the output4_edge_detection_contour. 
4. I am working on object outermost boundary identification. As the problem statement is the sequential solving of subproblems once this part is done everything can be solved easily. 

## Step 3: Barcode detection
1. barcode_bounding_box.py - Barcode decoding and detection is done using pyzbar library and output is stored in output5_barcode_detection_and_decode_with_pyzbar. 
2. all barcode_barcode_detection_output, missing and partial barcode_detection_output, missing barcode_detection_output, partial barcode_detection_output has output of barcode_detection_with_text_removal.py. Here noises due to texts are removed. 
3. Looking forward to a pre-trained model for barcode decoding because some barcodes are not able to decode even by specifically cropping the barcode when feeding the input. 
