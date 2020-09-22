from utils import create_input_files

if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(dataset='flickr8k',
                       karpathy_json_path=r"D:\Users\wt\Downloads\caption_datasets\dataset_flickr8k.json",
                       image_folder=r"D:\Users\wt\Downloads\image_caption\flickr8k\images",
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder=r"D:\Users\wt\Downloads\image_caption\outputfile",
                       max_len=50)

# try if I can commit!