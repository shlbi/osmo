import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description='OSMO reproducible PK research pipeline')
    parser.add_argument('--data-root',type=Path,default=Path('data'))
    parser.add_argument('--reports',type=Path,default=Path('reports'))
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('acquire');sub.add_parser('prepare')
    train=sub.add_parser('train-baseline')
    train.add_argument('--species',default='rat');train.add_argument('--tissue',default='plasma')
    train.add_argument('--trees',type=int,default=200)
    args=parser.parse_args()
    if args.command=='acquire':
        from .cvtdb import acquire
        result=acquire(args.data_root)
    elif args.command=='prepare':
        from .cvtdb import prepare
        result=prepare(args.data_root,args.reports)
    else:
        from .baseline import train
        result=train(args.data_root,args.reports,args.species,args.tissue,args.trees)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
