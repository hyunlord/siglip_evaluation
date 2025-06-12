from typer import Option, Typer
from functools import partial

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoModel

from app.util import image_collate_fn, make_text_features, make_image_features, calculate_recall

cmd = Typer()


@cmd.command()
def evaluation(
        model_path: str = Option(
            "google/siglip-base-patch16-224",
            "-m",
            "--model",
            help="path of model for evaluation",
            rich_help_panel="model"
        ),
        model_device: str = Option(
            "cuda",
            "-md",
            "--device",
            help="select device",
            rich_help_panel="model"
        ),
        image_batch_size: int = Option(
            16,
            "-ibs",
            "--image-batch-size",
            help="image batch size",
            rich_help_panel="model"
        ),
        text_batch_size: int = Option(
            256,
            "-tbs",
            "--text-batch-size",
            help="text batch size",
            rich_help_panel="model"
        ),
        data_path: str = Option(
            "hyunlord/aihub_image-caption_en-ko_karpathy-split",
            "-d",
            "--data",
            help="path of data for evaluation",
            rich_help_panel="data"
        ),
        data_type: str = Option(
            "test",
            "-dt",
            "--data-type",
            help="data type for evaluation",
            rich_help_panel="data"
        ),
        data_count: int = Option(
            0,
            "-dc",
            "--data-count",
            help="data count for evaluation",
            rich_help_panel="data"
        ),
        data_column: str = Option(
            'captions',
            "-dcol",
            "--data-column",
            help="data column for evaluation",
            rich_help_panel="data"
        )
):
    device = model_device
    print(f"   - 디바이스 : {device}")

    print(f"   - {model_path} || 모델 및 프로세서 로딩 중")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device)
    model.eval()
    print(f"   - 모델 및 프로세서 로딩 완료")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   - 모델 전체 파라미터 수: {total_params:,}")

    if data_count:
        print(f"   - {data_path} || {data_type}:{data_count} 데이터 로딩 중")
        dataset = load_dataset(data_path, split=f"{data_type}[:{data_count}]")
    else:
        print(f"   - {data_path} || {data_type} 데이터 로딩 중")
        dataset = load_dataset(data_path, split=f"{data_type}")
    print(f"   - 데이터 로딩 완료")
    image_to_text_map = [list(range(i * 5, (i + 1) * 5)) for i in range(len(dataset))]
    text_to_image_map = [i for i in range(len(dataset)) for _ in range(5)]

    all_captions = [caption for ex in dataset for caption in ex[data_column][:5]]
    print(f" 데이터셋 캡션 개수: {len(all_captions)}")

    collate_with_processor = partial(image_collate_fn, processor=processor, device=device)
    image_dataloader = DataLoader(dataset, batch_size=image_batch_size, collate_fn=collate_with_processor, shuffle=False)

    all_text_features = make_text_features(model, processor, all_captions, text_batch_size, device)
    all_image_features = make_image_features(model, image_dataloader, device)
    recall_at_1_i2t, recall_at_1_t2i = calculate_recall(model, all_text_features, all_image_features,
                                                        text_to_image_map, image_to_text_map)

    print("\n" + "=" * 50)
    print(f"   - Model Path : {model_path})")
    print(f"   - Data Path : {data_path}), Data Type : {data_type}, Data Count : f{data_count}")
    print(f"   - 최종 평가 결과")
    print("=" * 50)
    print(f"Image-to-Text Recall@1: {recall_at_1_i2t:.2f}%")
    print(f"Text-to-Image Recall@1: {recall_at_1_t2i:.2f}%")
    print("=" * 50)


if __name__ == '__main__':
    cmd()
