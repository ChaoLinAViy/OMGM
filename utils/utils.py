import csv
import os
from tqdm import tqdm
import PIL

GLD_image_path = "../../datasets/OMGM_data/E-VQA/landmarks" 
iNat_image_path = "../../datasets/OMGM_data/E-VQA/inaturalist"
infoseek_test_path = "../../datasets/OMGM_data/InfoSeek/infoseek_val_images"
infoseek_train_path = "../../datasets/OMGM_data/InfoSeek/infoseek_train_images"

def get_image(image_id, dataset_name):
    """
        get the image file by image_id. image id are indexed by its first 3 letters in the corresponding folder. e.g. image_id = "abcde" will be stored in "a/b/c/abcde.jpg"
    Args:
        image_id : the image id
    """
    if dataset_name == "inaturalist":
        image_path = os.path.join(iNat_image_path, image_id + ".jpg")
    elif dataset_name == "landmarks":
        image_path = os.path.join(GLD_image_path, image_id + ".jpg") #image_id[0], image_id[1], image_id[2],
    elif dataset_name == "infoseek":
        if os.path.exists(os.path.join(infoseek_test_path, image_id + ".jpg")):
            image_path = os.path.join(infoseek_test_path, image_id + ".jpg")
        elif os.path.exists(os.path.join(infoseek_test_path, image_id + ".JPEG")):
            image_path = os.path.join(infoseek_test_path, image_id + ".JPEG")
    else:
        raise NotImplementedError("dataset name not supported")
    return image_path
def get_train_image(image_id, dataset_name):
    """
        get the train image file by image_id. image id are indexed by its first 3 letters in the corresponding folder. e.g. image_id = "abcde" will be stored in "a/b/c/abcde.jpg"
    Args:
        image_id : the image id
    """
    if dataset_name == "E-VQA":
        if os.path.exists(os.path.join(iNat_image_path, image_id + ".jpg")):
            image_path = os.path.join(iNat_image_path, image_id + ".jpg")
        elif os.path.exists(os.path.join(GLD_image_path, image_id + ".jpg")):
            image_path = os.path.join(GLD_image_path, image_id + ".jpg")
        else:
            raise ValueError(f"Image {image_id} not found in {iNat_image_path} or {GLD_image_path}")
    elif dataset_name == "InfoSeek":
        if os.path.exists(os.path.join(infoseek_train_path, image_id + ".jpg")):
            image_path = os.path.join(infoseek_train_path, image_id + ".jpg")
        elif os.path.exists(os.path.join(infoseek_train_path, image_id + ".JPEG")):
            image_path = os.path.join(infoseek_train_path, image_id + ".JPEG")
        else:
            raise ValueError(f"Image {image_id} not found in {infoseek_train_path}")
    else:
        raise NotImplementedError("dataset name not supported")
    return image_path

def load_csv_data(test_file):
    test_list = []
    with open(test_file, "r") as f:
        reader = csv.reader(f)
        test_header = next(reader)
        for row in reader:
            try: 
                if (row[test_header.index("question_type")] == "automatic" or row[test_header.index("question_type")] == "templated" or row[test_header.index("question_type")] == "multi_answer" or row[test_header.index("question_type")] == "infoseek"): 
                    test_list.append(row)
            except:
                # print row and line number
                print(row, reader.line_num)
                raise ValueError("Error in loading csv data")
    return test_list, test_header

def get_title2imgpaths(wiki_img_url_file):
    title2imgpaths = {}
    for split in range(1, 14):
        wiki_img_url_path = wiki_img_url_file.format(split_num=split)
        with open(wiki_img_url_path, 'r') as wf:
            reader = csv.reader(wf)
            first_row = next(reader)
            for row in tqdm(reader, desc=f'add {wiki_img_url_path} to title2imgpaths'):
                title = row[0]
                img_path = row[2] # full/wiki_image_split/wiki_entity_image_{num}/image
                if title not in title2imgpaths:
                    title2imgpaths[title] = []
                if len(title2imgpaths[title]) >= 10 or img_path in title2imgpaths[title]:
                    continue
                title2imgpaths[title].append(img_path)
            wf.close()
        print(f'get title2imgpaths done. length:', len(title2imgpaths))
        del reader, first_row
    return title2imgpaths

def get_title2wikiimg(wiki_img_url_file, wiki_img_path_prefix):
    title2wikiimg = {} # wiki_title -> [{'caption': caption, 'img_path': img_path}]
    for split in tqdm(range(1, 14),desc = 'load wiki img csv'):
        wiki_img_url_path = wiki_img_url_file.format(split_num=split)
        with open(wiki_img_url_path, 'r') as wf:
            reader = csv.reader(wf)
            first_row = next(reader)
            for row in reader:
                title = row[0]
                img_path = row[2]
                caption = row[3]
                if title not in title2wikiimg:
                    title2wikiimg[title] = []
                image_path = os.path.join(wiki_img_path_prefix, img_path)
                
                if  not os.path.exists(image_path) or len(title2wikiimg[title]) >= 10 or {'caption': caption,'img_path': image_path} in title2wikiimg[title]:
                    continue
                title2wikiimg[title].append({'caption': caption,'img_path': image_path})
            wf.close()
    print(f'get title2wikiimg done. length of entities in it:', len(title2wikiimg))
    del reader, first_row
    return title2wikiimg

def get_title2wikiimg_addpix(wiki_img_url_file, wiki_img_path_prefix):
    title2wikiimg = {} # wiki_title -> [{'caption': caption, 'img_path': img_path}]
    for split in tqdm(range(1, 14),desc = 'load wiki img csv with pixels'):
        wiki_img_url_path = wiki_img_url_file.format(split_num=split)
        with open(wiki_img_url_path, 'r') as wf:
            reader = csv.reader(wf)
            first_row = next(reader)
            for row in reader:
                title = row[0]
                img_path = row[2]
                caption = row[3]
                if title not in title2wikiimg:
                    title2wikiimg[title] = []
                image_path = os.path.join(wiki_img_path_prefix, img_path)
                cur_captions = [x['caption'] for x in title2wikiimg[title]]
                if  not os.path.exists(image_path) or len(title2wikiimg[title]) >= 10 or caption in cur_captions:
                    continue
                title2wikiimg[title].append({'caption': caption,'img_path':os.path.join(wiki_img_path_prefix, img_path), 'image_pixels': PIL.Image.open(image_path).convert("RGB")})
            wf.close()
    print(f'get title2wikiimg done. length of entities in it:', len(title2wikiimg))
    del reader, first_row
    return title2wikiimg

    


def get_test_question(preview_index, test_list, test_header):
    return {test_header[i]: test_list[preview_index][i] for i in range(len(test_header))}

def remove_list_duplicates(test_list):
    # remove duplicates
    seen = set()
    return [x for x in test_list if not (x in seen or seen.add(x))]
    