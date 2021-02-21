import os
import click
from PIL import Image
from glob import glob
from tqdm import tqdm


@click.command()
@click.argument('data_dir', type=click.Path())
@click.option('--width', '-w', type=int, default=512)
@click.option('--height', '-h', type=int, default=512)
def standardize_dataset_res(data_dir, width=512, height=512):
    """ Given directory of image dataset, resize all to new_res
        and store in new folder with _wxh appended"""
    
    ### Need to copy the class_dict as well... ###
    
    parent_folder, dataset = os.path.split(data_dir)
    new_dataset = dataset+'_{}x{}'.format(width,height)
    new_data_dir = os.path.join(parent_folder, new_dataset)
    all_imgs = glob(os.path.join(data_dir, '**/*.png'))

    for p in tqdm(all_imgs):
        img = Image.open(p)
        img = img.resize((height,width))
        new_path = p.replace(dataset, new_dataset)
        os.makedirs(os.path.split(new_path)[0], exist_ok=True)
        # print(p)
        # print(dataset)
        # print(new_dataset)
        img.save(p.replace(dataset, new_dataset))

    return


if __name__ == '__main__':
    standardize_dataset_res()
