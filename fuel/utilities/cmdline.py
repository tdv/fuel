import argparse

ARG_SOURCE = "source"
SOURCE_FILE = "file"
SOURCE_WEBCAM = "webcam"

ARG_SRC_IMAGE_FILE = "input_image_file"
ARG_SRC_WEBCAM = "webcam"

ARG_DESTINATION = "destination"
DEST_FAKE_WEBCAM = "fakecam"
DEST_IMAGE_FILE = "file"
DEST_WINDOW = "window"

ARG_DEST_IMAGE_FILE = "output_image_file"

ARG_DEST_FAKECAM = "fakecam"

ARG_MODE = "mode"
MODE_BLUR = "blur"
MODE_BACKGRAUND = "background"

BLUR_KERNEL_SIZE = "blur_kernel_size"

BACKGROUND_FILENAME = "background"

def get_args():
    parser = argparse.ArgumentParser(description="Background substitutino or blur arguments",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-s", "--{}".format(ARG_SOURCE),
                        help="input source: {} | {}".format(SOURCE_WEBCAM, SOURCE_FILE),
                        default=SOURCE_WEBCAM)
    parser.add_argument("-i", "--{}".format(ARG_SRC_IMAGE_FILE),
                        help="path to an image file (only for 'file' source)",
                        default=None)
    parser.add_argument("-w", "--{}".format(ARG_SRC_WEBCAM),
                        help="path to real webcam (only for 'webcam' source)",
                        default="/dev/video0")
    parser.add_argument("-d", "--{}".format(ARG_DESTINATION),
                        help="output destination: {0} | {1} | {2} ('{2}' means opencv window) "
                        .format(DEST_FAKE_WEBCAM, DEST_IMAGE_FILE, DEST_WINDOW),
                        default=DEST_FAKE_WEBCAM)
    parser.add_argument("-f", "--{}".format(ARG_DEST_FAKECAM),
                        help="path to fake webcam (example: /dev/video3)",
                        default="/dev/video3")
    parser.add_argument("-o", "--{}".format(ARG_DEST_IMAGE_FILE),
                        help="output file path",
                        default=None)
    parser.add_argument("-m", "--{}".format(ARG_MODE),
                        help="mode: {} or {} substitution (blur|background)"
                        .format(MODE_BLUR, MODE_BACKGRAUND),
                        default=MODE_BLUR)
    parser.add_argument("-k", "--{}".format(BLUR_KERNEL_SIZE),
                        help="gaussianblur kernel size (only for 'blur' mode)",
                        default="55")
    parser.add_argument("-b", "--{}".format(BACKGROUND_FILENAME),
                        help="background file path (only for 'background' mode)",
                        default=None)
    args = parser.parse_args()
    return vars(args)