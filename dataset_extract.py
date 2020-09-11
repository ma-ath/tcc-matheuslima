import argparse, sys
from include.globals_and_functions import *
from include.telegram_logger import telegramSendMessage

def main(argv):
    #   Parse command line to read with which network should the script extract image information
    parser = argparse.ArgumentParser(description='Receives the desired network model for extraction')
    parser.add_argument('--network','-n', default="vgg16", help='The desired network')
    args = parser.parse_args()
    
    #   Check if network is suported
    #   We now only have vgg16
    if args.network != "vgg16":
        print_error("We currently only suport the VGG16 model")
        exit(1)

    
if __name__ == "__main__":
    try:
        main(sys.argv)

    except Exception as e:
        print_error('An error has occurred')
        print_error(str(e))
        telegramSendMessage('[ERROR]: An error has occurred')
        telegramSendMessage(str(e))