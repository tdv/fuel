import argparse

SOURCE_FILE = "file"
SOURCE_WEBCAM = "webcam"

DEST_FAKE_WEBCAM = "fakecam"
DEST_IMAGE_FILE = "file"
DEST_WINDOW = "window"

MODE_BLUR = "blur"
MODE_BACKGRAUND = "background"

def get_args():
    parser = argparse.ArgumentParser(description="Background substitutino or blur arguments",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-s", "--source",
                        help="input source: {} | {}".format(SOURCE_WEBCAM, SOURCE_FILE),
                        default=SOURCE_WEBCAM)
    parser.add_argument("-i", "--input_image_file",
                        help="path to an image file (only for 'file' source)",
                        default=None)
    parser.add_argument("-w", "--webcam",
                        help="path to real webcam (only for 'webcam' source)",
                        default="/dev/video0")
    parser.add_argument("-d", "--destination",
                        help="output destination: {0} | {1} | {2} ('{2}' means opencv window) "
                        .format(DEST_FAKE_WEBCAM, DEST_IMAGE_FILE, DEST_WINDOW),
                        default=DEST_FAKE_WEBCAM)
    parser.add_argument("-f", "--fakecam",
                        help="path to fake webcam (example: /dev/video3)",
                        default="/dev/video3")
    parser.add_argument("-o", "--output_image_file",
                        help="output file path",
                        default=None)
    parser.add_argument("-m", "--mode",
                        help="mode: {} or {} substitution (blur|background)"
                        .format(MODE_BLUR, MODE_BACKGRAUND),
                        default=MODE_BLUR)
    parser.add_argument("-k", "--blur_kernel_size",
                        help="gaussianblur kernel size (only for 'blur' mode)",
                        default=55, type=int)
    parser.add_argument("-b", "--background",
                        help="background file path (only for 'background' mode)",
                        default=None)
    return parser.parse_args()