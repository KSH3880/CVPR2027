import os
from smplx import SMPL, SMPLX, SMPLH

def get_body_model(type, gender, batch_size, debug=True):
    # 统一管理人体模型, 防止出现错用导致生成的motion不一致
    if type == "SMPL":
        if debug:
            print()
            print("CAUTION: You are using **gender={}** SMPL model!".format(gender))
            print("CAUTION: You are using **gender={}** SMPL model!".format(gender))
            print("CAUTION: You are using **gender={}** SMPL model!".format(gender))
            print()
        body_model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "smpl")
        model = SMPL(
            model_path=body_model_dir,
            gender=gender,
            batch_size=batch_size,
            num_betas=10,
        )
    elif type == "SMPLX":
        if debug:
            print()
            print("CAUTION: You are using **gender={}** **flat_hand_mean=True** **use_pca=False** SMPL-X model!".format(gender))
            print("CAUTION: You are using **gender={}** **flat_hand_mean=True** **use_pca=False** SMPL-X model!".format(gender))
            print("CAUTION: You are using **gender={}** **flat_hand_mean=True** **use_pca=False** SMPL-X model!".format(gender))
            print()
        body_model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "smplx")
        model = SMPLX(
            model_path=body_model_dir,
            gender=gender,
            batch_size=batch_size,
            num_betas=10,
            use_pca=False,
            flat_hand_mean=True
        )
    elif type == "SMPLH":
        if debug:
            print()
            print("CAUTION: You are using **gender={}** **flat_hand_mean=True** **use_pca=False** SMPL-H model (16 betas)!".format(gender))
            print("CAUTION: You are using **gender={}** **flat_hand_mean=True** **use_pca=False** SMPL-H model (16 betas)!".format(gender))
            print("CAUTION: You are using **gender={}** **flat_hand_mean=True** **use_pca=False** SMPL-H model (16 betas)!".format(gender))
            print()
        body_model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "smplh")
        model = SMPLH(
            model_path=body_model_dir,
            gender=gender,
            batch_size=batch_size,
            num_betas=16,
            use_pca=False,
            flat_hand_mean=True
        )
    else:
        raise NotImplementedError

    return model
