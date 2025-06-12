import os
import random
import numpy as np
from tqdm import tqdm

import torch


def set_seed(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def image_collate_fn(batch, processor, device):
    valid_images = []
    for item in batch:
        image = item.get('image')
        if image is None:
            continue
        if image.mode != "RGB":
            image = image.convert("RGB")
        valid_images.append(image)
    if not valid_images:
        return None
    image_inputs = processor(images=valid_images, return_tensors="pt", padding=True).to(device)
    return {"pixel_values": image_inputs.pixel_values}


def make_text_features(model, processor, texts, batch_size, device):
    # 텍스트는 양이 많으므로 한 번에 처리
    with torch.no_grad():
        print("   - Entire Text Feature 추출...")
        text_inputs = processor(text=texts,
                                padding="max_length",
                                max_length=64,
                                truncation=True,
                                return_tensors="pt").to(device)
        text_features_list = []
        for i in tqdm(range(0, text_inputs.input_ids.shape[0], batch_size), desc="     텍스트 배치 처리"):
            batch_inputs = {k: v[i:i+batch_size].to(device) for k, v in text_inputs.items()}
            text_embeds = model.get_text_features(**batch_inputs)
            text_features_list.append(text_embeds)
        all_text_features = torch.cat(text_features_list)
        print("   - Entire Text Feature 준비 완료")
    return all_text_features


def make_image_features(model, image_dataloader, device):
    all_image_features = []
    with torch.no_grad():
        print("   - Entire Image Feature 추출...")
        for batch in tqdm(image_dataloader, desc="   이미지 배치 처리"):
            image_embeds = model.get_image_features(pixel_values=batch['pixel_values'].to(device))
            all_image_features.append(image_embeds)
        all_image_features = torch.cat(all_image_features)
        print("   - Entire Image Feature 준비 완료")
    return all_image_features


def calculate_recall(model_path, model, all_text_features, all_image_features, text_to_image_map, image_to_text_map):
    with torch.no_grad():
        all_image_features /= all_image_features.norm(p=2, dim=-1, keepdim=True)
        all_text_features /= all_text_features.norm(p=2, dim=-1, keepdim=True)

        logits_per_text = torch.matmul(all_text_features, all_image_features.T)
        if 'clip' in model_path:
            logits_per_text = logits_per_text * model.logit_scale.exp()
            logits_per_image = logits_per_text.t()
            probs = torch.softmax(logits_per_image, dim=1)
        else:
            logits_per_text = logits_per_text * model.logit_scale.exp() + model.logit_bias
            logits_per_image = logits_per_text.t()
            probs = torch.sigmoid(logits_per_image)
        pred_i2t = probs.argmax(dim=1)
        correct_i2t = sum(p in it_map for p, it_map in zip(pred_i2t, image_to_text_map))
        recall_at_1_i2t = (correct_i2t / len(all_image_features)) * 100

        pred_t2i = probs.argmax(dim=0)
        correct_t2i = sum(p == t_map for p, t_map in zip(pred_t2i, text_to_image_map))
        recall_at_1_t2i = (correct_t2i / len(all_text_features)) * 100
    return recall_at_1_i2t, recall_at_1_t2i
