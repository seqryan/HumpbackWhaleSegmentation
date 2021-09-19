import sys
import logging
import os
import argparse

import whale_detector as wd


# use level as logging.DEBUG for detailed logs and ogging.INFO for importnat logs
logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s] %(asctime)s - %(message)s', datefmt='%H:%M:%S')


def check_annotations(dataset_dir, annotations_file):
    train_annotation_path = os.path.join(dataset_dir, 'train', annotations_file)
    if not os.path.isfile(train_annotation_path):
        logging.error('Missing train annotation file \'%s\' in \'%s\' directory' %
                      (annotations_file, os.path.join(dataset_dir, 'train')))
        return False

    validaiton_annotation_path = os.path.join(dataset_dir, 'test', annotations_file)
    if not os.path.isfile(validaiton_annotation_path):
        logging.error('Missing validation annotation file \'%s\' in \'%s\' directory' %
                      (annotations_file, os.path.join(dataset_dir, 'test')))
        return False

    return True


def train_model(args):
    dataset_dir = args.dataset
    annotations_file = args.annotations_file_name
    weights_dir = args.save_weights

    if check_annotations(dataset_dir, annotations_file):
        # start training
        logging.info('Starting training ...')
        metadata = wd.register_dataset(dataset_dir, annotations_file)
        configuration = wd.train(weights_dir)
        # wd.infer(dataset_dir, annotations_file, configuration, metadata)
        logging.info('Training completed')


def load_and_save(args):
    weights_dir = args.load_weights
    dataset_dir = args.dataset
    output_dir = args.output
    annotations_file = args.annotations_file_name

    if check_annotations(dataset_dir, annotations_file):
        if os.path.isdir(output_dir):
            logging.info('Deleting previously found %s output directory' % output_dir)
            os.rmdir(output_dir)
        os.mkdir(output_dir)

        metadata = wd.register_dataset(dataset_dir, annotations_file)
        configuration = wd.load_model(weights_dir)
        wd.save_segmented_images(dataset_dir, output_dir, configuration, metadata)
        # wd.infer(configuration, metadata)  # uncomment to preview segmentation result


if __name__ == '__main__':
    # top-level parser
    parser = argparse.ArgumentParser(description="Humpback Whale Segmentation Runner")
    subparsers = parser.add_subparsers()
    subparsers.required = True

    # parser for the "train" command
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--save_weights', '-s', type=str, metavar='SAVE_WEIGHTS_DIR',
                              help='directory where the model weights will be saved', required=True)
    parser_train.add_argument('--dataset', '-d', type=str,  metavar='DATASET_DIR',
                              help='directory containing the train and test directories with images',
                              required=True)
    parser_train.add_argument('--annotations_file_name', '-a', type=str, metavar='ANNOTATIONS_FILE_NAME',
                              help='name of the annotations json file used to train the segmentation model',
                              required=True)
    parser_train.set_defaults(func=train_model)

    # parser for the "save" command
    parser_save = subparsers.add_parser('save')
    parser_save.add_argument('--load_weights', '-l', type=str, metavar='LOAD_WEIGHTS_DIR',
                             help='directory from where the model weights will be loaded', required=True)
    parser_save.add_argument('--dataset', '-d', type=str,  metavar='DATASET_DIR',
                             help='directory containing the train and test directories with images',
                             required=True)
    parser_save.add_argument('--output', '-o', type=str,  metavar='OUTPUT_DIR',
                             help='directory where the segmented train and test images will be saved', required=True)
    parser_save.add_argument('--annotations_file_name', '-a', type=str, metavar='ANNOTATIONS_FILE_NAME',
                             help='name of the annotations json file used to train the segmentation model',
                             required=True)
    parser_save.set_defaults(func=load_and_save)

    if len(sys.argv) == 1:
        parser.print_help()
    else:
        args = parser.parse_args(sys.argv[1:])
        args.func(args)
