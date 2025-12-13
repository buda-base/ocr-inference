from Config import MODEL_DICT


from BDRC.utils import download_model


def download_all_models():
    print(f"Available Models: {list(MODEL_DICT.keys())}")

    for k, v in MODEL_DICT.items():
        try:
            ret = download_model(v)
            print(ret)
        except BaseException as e:
            print(f"Failed to download model: {v}")


if __name__ == "__main__":

    download_all_models()