import argparse

ARG_SOURCE = "source"
SOURCE_FILE = "file"
SOURCE_WEBCAM = "webcam"

ARG_SRC_IMAGE_FILE = "input-image-file"
ARG_SRC_WEBCAM = "webcam"

ARG_DESTINATION = "destination"
DEST_FAKE_WEBCAM = "fakecam"
DEST_IMAGE_FILE = "file"
DEST_WINDOW = "window"

ARG_DEST_IMAGE_FILE = "output-image-file"

ARG_DEST_FAKECAM = "fakecam"

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
    parser.add_argument("-m", "--mode",
                        help="mode: blur or background substitution (blur|background)",
                        default="blur")
    parser.add_argument("-k", "--blur-kernel-size",
                        help="gaussianblur kernel size (only for 'blur' mode)",
                        default="55")
    parser.add_argument("-b", "--background",
                        help="background file path (only for 'background' mode)",
                        default=None)
    args = parser.parse_args()
    return vars(args)