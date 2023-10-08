import matplotlib.pyplot as plt
from tensorflow import keras
import os
import data
from glob import glob
from keras.models import load_model
import model
import argparse
from keras.callbacks import History
import patoolib

root = r"/user/HS402/sg02064/dissertation"

def parser():
    parser_var = argparse.ArgumentParser(description='MirNet')
    parser_var.add_argument('--data-dir', type=str)
    arg = parser_var.parse_args()
    return arg


def set_images(arg):
   
    root = arg.data_dir
    # if not os.path.exists('dataset'):
    #     patoolib.extract_archive(root)
# dataset/train/originals
    # patoolib.extract_archive(root)
    root = r"/user/HS402/sg02064/dissertation/"
    
    train_low_light_images = sorted(glob( os.path.join(root,r"dataset/train/projections/*")))
    train_enhanced_images = sorted(glob( os.path.join(root,r"dataset/train/originals/*")))

    val_low_light_images = sorted(glob( os.path.join(root,r"dataset/val/projections/*")))
    val_enhanced_images = sorted(glob( os.path.join(root,r"dataset/val/originals/*")))
        
    # val_enhanced_images = sorted(glob(r"D:\F Drive\Shubham\surrey\data- ct\dataset\val\poriginals\*"))
    print('train_low_light_images', len(train_low_light_images))
    print('train_enhanced_images',len(train_enhanced_images))
    print('val_low_light_images', len(val_low_light_images))
    print('val_enhanced_images',len(val_enhanced_images))

    train_dataset = model.get_dataset(train_low_light_images, train_enhanced_images)
    val_dataset = model.get_dataset(val_low_light_images, val_enhanced_images)

    print("Train Dataset:", train_dataset)
    print("Val Dataset:", val_dataset)
    return train_dataset, val_dataset


def define_mir_model():
    mir = model.mirnet_model(num_rrg=3, num_mrb=2, channels=64)
    # **********************************************************************************************************************

    optimizer = keras.optimizers.Adam(learning_rate=1e-4)
    mir.compile(
        optimizer=optimizer, loss=model.charbonnier_loss, metrics=[model.peak_signal_noise_ratio,
                                                                   model.structural_similarity]
    )
    return mir


def fit_mir_model(mir, train_dataset, val_dataset):
    saver = model.CustomSaver()

    history_callback = model.HistorySaver()

    custom_objects= {'charbonnier_loss': model.charbonnier_loss,
                 'peak_signal_noise_ratio': model.peak_signal_noise_ratio,
                 'structural_similarity' : model.structural_similarity}

    model_loaded = load_model(r"/user/HS402/sg02064/dissertation/model_files/model_25.h5",custom_objects= custom_objects )
    optimizer = keras.optimizers.Adam(learning_rate=1e-4)
    model_loaded.compile(
        optimizer=optimizer, loss=model.charbonnier_loss, metrics=[model.peak_signal_noise_ratio,
                                                                   model.structural_similarity]
    )
    history = model_loaded.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=25,
        callbacks=[
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_peak_signal_noise_ratio",
                factor=0.5,
                patience=5,
                verbose=1,
                min_delta=1e-7,
                mode="max",
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_structural_similarity",
                factor=0.5,
                patience=5,
                verbose=1,
                min_delta=1e-7,
                mode="max",
            ),
            saver
            , 
            history_callback

        ],
    )
    return history

def loss_graph(root, history):
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Train and Validation Losses Over Epochs", fontsize=14)
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(root, 'results/loss.png'))

def psnr_graph(root, history):
    plt.plot(history.history["peak_signal_noise_ratio"], label="train_psnr")
    plt.plot(history.history["val_peak_signal_noise_ratio"], label="val_psnr")
    plt.xlabel("Epochs")
    plt.ylabel("PSNR")
    plt.title("Train and Validation PSNR Over Epochs", fontsize=14)
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(root, 'results/psnr.png'))
    

def graphs(history):
    root = r"/user/HS402/sg02064/dissertation"
    loss_graph(root= root,history= history)

    psnr_graph(root, history)

    plt.plot(history.history["peak_signal_noise_ratio"], label="train_psnr")
    plt.plot(history.history["val_structural_similarity"], label="val_ssim")
    # plt.plot(history.history["val_peak_signal_noise_ratio"], label="val_psnr")
    plt.xlabel("Epochs")
    plt.ylabel("SSIM")
    plt.title("Train and Validation SSIM Over Epochs", fontsize=14)
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(root, 'results/ssim.png'))


if __name__ == '__main__':
    # os.mkdir('dataset/')

    arg = parser()
    # data.process()

    # model.infer(r'/user/HS402/sg02064/dissertation/dataset/val/projections/img_4_64.jpeg')

    # # **********************************************************************************************************************
    # # train_root = r'D:\F Drive\Shubham\surrey\data- ct\dataset\train\originals'
    # # train_destination = r'D:\F Drive\Shubham\surrey\data- ct\dataset\train\projections'
    # # print('train data being processed')
    # # data.process(train_root, train_destination)
    # #
    # # val_root = r'D:\F Drive\Shubham\surrey\data- ct\dataset\val\originals'
    # # val_destination = r'D:\F Drive\Shubham\surrey\data- ct\dataset\val\projections'
    # # print('test data being processed')
    # # data.process(val_root, val_destination)
    #
    # # **********************************************************************************************************************
    
    train_dataset, val_dataset = set_images(arg)
    mir = define_mir_model()
    history = fit_mir_model(mir, train_dataset, val_dataset)
    graphs(history)
