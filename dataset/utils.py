import nlpaug.augmenter.word as naw
import argparse
import codecs
from pathlib import Path
import re

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-infile', type=str, default=r'C:\project\nlp_rep\dataset\en-ud-train.txt', help='UD file to augment')
    parser.add_argument('-outdir', type=str, default=r'.', help='Output dir')
    parser.add_argument('-num_swaps', type=int, default=2, help='number of new sentences from each sentence')
    config = parser.parse_args()
    return config


def main(config):
    infile = Path(config.infile)
    if not infile.is_file():
        raise FileNotFoundError
    pattern = re.compile('(train|test|val|dev).txt')
    phase = pattern.findall(infile.name)[0]

    fin = codecs.open(infile, 'r', 'utf-8')
    txt = fin.read()
    fin.close()

    outdir = Path(config.outdir)
    if not outdir.is_dir():
        outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / f'randswap_aug_{phase}.txt'
    fout = codecs.open(outfile, 'w', 'utf-8')

    aug = naw.RandomWordAug(action='swap')
    lines = []
    for line in txt.split('\n'):
        lines.append(line)
        for _ in range(config.num_swaps):
            augmented_text = aug.augment(line)
            lines.append(augmented_text)
    fout.writelines(f"{line}\n" for line in lines)
    fout.close()


if __name__ == '__main__':
    config = parse()
    main(config)
