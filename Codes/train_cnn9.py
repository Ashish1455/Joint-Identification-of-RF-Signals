import os
import argparse
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard

from cnn9_model import build_cnn9
from Codes.data_prep_cnn9 import load_dataset, make_datasets


def setup_tf(cpu_only: bool = False, allow_growth: bool = True):
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if cpu_only:
            tf.config.set_visible_devices([], 'GPU')
            print('[TF] Using CPU only.')
            return
        if gpus and allow_growth:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except Exception:
                    pass
            print(f"[TF] Enabled memory growth on {len(gpus)} GPU(s).")
    except Exception as e:
        print(f"[TF] Setup warning: {e}")


def main(args):
    setup_tf(cpu_only=args.cpu, allow_growth=args.allow_growth)

    x, y, dims = load_dataset(args.mat)
    C, M, S = dims
    num_classes = int(C * M)

    model = build_cnn9(input_shape=x.shape[1:], num_classes=num_classes)
    optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr, momentum=0.9)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    print(f"[Data] X: {x.shape}, y: {y.shape}, classes: {num_classes} (C={C}, M={M}, S={S} ignored in labels)")
    ds_tr, ds_val, ds_test = make_datasets(
        x, y,
        val_split=args.val_split,
        test_split=args.test_split,
        shuffle=True,
        seed=42,
        batch_size=args.batch_size,
    )

    os.makedirs(args.out, exist_ok=True)
    ckpt = ModelCheckpoint(os.path.join(args.out, 'best_cnn9.h5'), monitor='val_accuracy', save_best_only=True, save_weights_only=False)
    rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-6)
    tb = TensorBoard(log_dir=os.path.join(args.out, 'tb'))

    history = model.fit(ds_tr, validation_data=ds_val, epochs=args.epochs, callbacks=[ckpt, rlrop, tb])

    model.save(os.path.join(args.out, 'last_cnn9.h5'))
    test_metrics = model.evaluate(ds_test, return_dict=True)
    print('[Test] metrics:', test_metrics)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mat', type=str, default='dataset_all1.mat')
    p.add_argument('--out', type=str, default='cnn9_runs')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-2, help='Initial learning rate')
    p.add_argument('--val_split', type=float, default=0.1)
    p.add_argument('--test_split', type=float, default=0.1)
    p.add_argument('--cpu', action='store_true', help='Force CPU-only training')
    p.add_argument('--allow_growth', action='store_true', default=True, help='Enable GPU memory growth')
    args = p.parse_args()
    main(args)
