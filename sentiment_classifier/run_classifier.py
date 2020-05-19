from fastai.text import *
import sentiment_classifier.config as cfg


model = {'transformer': Transformer,
         'awd_lstm': AWD_LSTM,
         'transformer_xl': TransformerXL}


def download_data():
    data_lm = (TextList.from_folder(cfg.path)
               # Inputs: all the text files in path
               .filter_by_folder(include=['train', 'test', 'unsup'])
               # We randomly split and keep 10% (10,000 reviews) for validation
               .split_by_rand_pct(0.1, seed=42)
               # We want to make a language model so we label accordingly
               .label_for_lm()
               .databunch(bs=cfg.batch_size))
    data_clas = (TextList.from_folder(cfg.path, vocab=data_lm.vocab)
                 # split by train and valid folder (that only keeps 'train' and 'test' so no need to filter)
                 .split_by_folder(valid='test')
                 # label them all with their folders
                 .label_from_folder(classes=['neg', 'pos'])
                 .databunch(bs=cfg.batch_size))
    data_lm.save(cfg.output_dir / 'data_lm_export.pkl')
    data_clas.save(cfg.output_dir / 'data_clas_export.pkl')


def train_lm(data_lm):
    learn = language_model_learner(data_lm, model[cfg.model_name], drop_mult=0.3)
    learn.fit_one_cycle(1, cfg.lm_lr0)
    learn.unfreeze()
    learn.fit_one_cycle(2, cfg.lm_lr1)
    learn.save_encoder(cfg.output_dir / f'ft_enc_{cfg.model_name}')


def train_sa(learn):
    learn.load_encoder(cfg.output_dir / f'ft_enc_{cfg.model_name}')
    learn.fit_one_cycle(1, cfg.cls_lr0)
    learn.unfreeze()
    learn.fit_one_cycle(2, slice(cfg.cls_lr1/100, cfg.cls_lr1))
    learn.save(cfg.output_dir / f'sa_clas_{cfg.model_name}')


def main():
    if cfg.download_data:
        download_data()
    data_lm = load_data(cfg.path, cfg.output_dir / 'data_lm_export.pkl')
    data_cls = load_data(cfg.path, cfg.output_dir / 'data_clas_export.pkl')

    # fine tune lm
    if cfg.train_lm:
        train_lm(data_lm)

    # fine tune classifier
    learn = text_classifier_learner(data_cls,  model[cfg.model_name], drop_mult=0.5)
    if cfg.train_sa:
        train_sa(learn)
    else:
        learn.load(cfg.output_dir / f'sa_clas_{cfg.model_name}')

    # predict
    learn.validate(dl=data_cls.test_dl)


if __name__ == '__main__':
    main()
