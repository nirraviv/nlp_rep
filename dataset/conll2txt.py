import argparse
from pathlib import Path
import codecs
import re


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-infile', type=str, default=r'C:\project\crop-rotate-augment\data\ud-treebanks-v2.1\UD_English\treeswap_aug_train.conllu', help='UD file to augment')
    parser.add_argument('-outdir', type=str, default=r'.', help='Output file')
    parser.add_argument('-re', default=False)
    config = parser.parse_args()
    return config


def main(config):
    infile = Path(config.infile)
    if not infile.is_file():
        raise FileNotFoundError
    outdir = Path(config.outdir)
    if not outdir.is_dir():
        outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / infile.name.replace(infile.suffix, '.txt')
    fout = codecs.open(outfile, 'w', 'utf-8')

    fin = codecs.open(infile, 'r', 'utf-8')
    lines = fin.read()
    fin.close()

    get_text = re.compile('# text = (.*?)\n')
    sentences = lines.split('\n\n')
    lines = []
    for sent in sentences:
        if config.re:
            txt = get_text.findall(sent)
            txt = txt[0]
        else:
            txt = str()
            words = sent.split('\n')
            if len(words) <=1: continue
            for i, line in enumerate(words):
                tmp = line.split('\t')
                txt += f'{tmp[1]} ' if not tmp[-1].startswith('Space') and i < len(words)-1 else tmp[1]
        if len(txt):
            lines.append(txt)
    fout.writelines(f"{line}\n" for line in lines)

    fout.close()
    pass


if __name__ == '__main__':
    config = parse()
    main(config)